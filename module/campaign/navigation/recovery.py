"""主线异常后的独立镜头回正，不使用失败规划的坐标，也不发送小队移动。"""

import cv2
import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_endpoints, pan_scene
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.perception import detect_markers, squad_arrow
from module.campaign.vision.surface_motion import project
from module.logger import logger

# 横向视野更长，左右各搜索四次；每次拖动端点仍在安全场景区域内。
RECOVERY_DRAGS = ((1100., 0.),) * 4 + ((-1100., 0.),) * 4 + ((0., 650.),) * 2 + ((0., -650.),) * 2
CENTER_TOLERANCE = 50.
GROUND_STABLE_TOLERANCE = 20.
CENTER_MIN_PROGRESS = 12.
CENTER_STALL_LIMIT = 2
CENTER_ADJUSTMENT_LIMIT = 8
VERIFY_INTERVAL = .4


def centering_frame(session):
    """取得同一紧凑帧的小队箭头、圆环与离屏标记，不依赖道路配准或历史位置。

    圆环坐标属于缩放至 1920×1080 后的紧凑 ROI，与 controls.GAZE 同源；箭头换算的
    地面点属于原始客户区，只用于镜头调整及按下点避让，不作为小队移动的落点。
    圆环不唯一时按缺失处理；截图前检查取消、窗口和时限，弹窗由身份闸门向上传播。
    """
    session.check()
    field = session.win.client()
    session.identity(field)
    arrow = squad_arrow(field)
    panel = cv2.resize(field, (1920, 1080))[96:307, 25:243]
    players, _ = detect_markers(panel, np.eye(3), mode='compact')
    ground = None if arrow is None else np.asarray(arrow, float) + [0., goto.ARROW_GROUND_OFFSET]
    evidence = dict(ground=ground, ring=players[0] if len(players) == 1 else None,
                    offscreen=squad_offscreen(field))
    session.record('recovery_center_frame', field=field, **evidence)
    return evidence


def centered(evidence):
    """只按箭头地面点判断小队是否位于视野中央附近。

    离屏图标出现、箭头缺失或坐标非有限值时不通过；小地图圆环不参与回正判定。
    """
    ground = evidence['ground']
    if evidence['offscreen'] is not None or ground is None:
        return False
    ground = np.asarray(ground, float)
    return bool(np.isfinite(ground).all()
                and np.linalg.norm(ground - goto.VIEWPORT_PIXEL) <= CENTER_TOLERANCE)


def centered_compact_ring(evidence):
    """返回镜头注视点附近的实测小队圆环，否则返回 None。

    只用于图标回正后或短拖未能修正箭头位置时的独立复核。环坐标属于归一化的紧凑面板；
    调用方须用连续帧检查环未移动，之后仍需展开配准确认小队的底图坐标。
    """
    ring = np.asarray(evidence['ring'], float)
    if evidence['offscreen'] is not None or ring.shape != (2,) or not np.isfinite(ring).all():
        return None
    return ring if np.linalg.norm(ring - goto.GAZE) <= 5. else None


def verify_centered_compact_ring(session, evidence):
    """按同一次会话的两帧紧凑小地图复核唯一圆环，只返回稳定的注视点附近圆环。

    失败时保留最后一帧供调用方继续辨认箭头和离屏图标。圆环坐标为归一化面板像素；
    镜头居中证据仍须结合之后新取得的小队地图定位使用。
    """
    ring = centered_compact_ring(evidence)
    if ring is None:
        return evidence, None
    session.pause(VERIFY_INTERVAL)
    checked = centering_frame(session)
    checked_ring = centered_compact_ring(checked)
    if checked_ring is not None and np.linalg.norm(checked_ring - ring) <= 2.:
        return checked, checked_ring
    return checked, None


def recenter_mechanism_edge(session, mechanism_position):
    """已关联中央解锁光幕后，按机关位置与新视口给边缘小队执行一次有界镜头短拖。

    仅用于机关触发后箭头和离屏图标同时缺失的场景。再次展开地图独立配准视口，将机关
    标注投回面板；落点位于面板内部安全区时拒绝推定小队在边缘。拖动方向由固定镜头
    仿射给出，最大 280 客户区像素，按下和抬起都限制在中央区域以避开边缘小队。
    不点击地面，不以投影位置作为导航小队坐标；拖动后仍走原箭头或离屏图标回正及实测定位。
    配准、标注、窗口或输入失败向上传播，普通恢复不调用本方法。
    """
    position = np.asarray(mechanism_position, float)
    if (position.shape != (2,) or not np.isfinite(position).all()
            or np.any(position < 0) or np.any(position >= session.owner.state.size)):
        raise ValueError('机关边缘回正缺少有效机关标注位置。')
    report = session.mechanism_viewport()
    matrix = np.asarray(report['roi_to_map'], float)
    panel_point = project(np.linalg.inv(matrix), position)
    if (not np.isfinite(panel_point).all()
            or (32 <= panel_point[0] <= 486 - 32 and 32 <= panel_point[1] <= 462 - 32)):
        raise ValueError('机关位置未处于展开地图视野边缘，不能执行边缘小队回正。')
    offset = -goto.GROUND_TO_CLIENT @ (position - np.asarray(report['position'], float))
    distance = float(np.linalg.norm(offset))
    if not np.isfinite(distance) or distance < 8:
        raise ValueError('机关边缘回正没有有效镜头拖动方向。')
    drag = offset * min(1., 280. / distance)
    start, end = pan_endpoints((500, 280, 1275, 760), drag)
    goto.map_close(session.win)
    session.identity(session.win.client())
    session.check()
    origin = np.asarray(session.win.client_origin(), int)
    session.win.handler.mouse_swipe(tuple(origin + start), tuple(origin + end))
    if session.win.handler._failures:
        raise RuntimeError('机关边缘回正镜头短拖失败，未发送地面点击。')
    session.record('mechanism_edge_recovery', force=True, mechanism_position=position.tolist(),
                   viewport=report['position'], panel_point=panel_point.tolist(), drag=drag.tolist())
    session.pause(1.2)


def recenter_squad(session, mechanism_position=None, *, accept_centered_ring=False):
    """短拖按箭头居中，长拖只用于寻找并点击离屏小队图标。

    初始箭头可见时按地面点短拖，双帧均以箭头居中且位移不超过 20 像素为准。短拖连续无效或
    达到总次数上限后改为定向搜索，避免镜头到达地图边缘后反复拖同一方向。需要定向
    搜索时在场景安全区域按右、左、下、上长拖，途中不再按箭头微调；离屏图标出现后立即点击
    并直接视为回正成功，不再复核。普通回正不展开地图，定向搜索用尽后抛 ValueError。
    回正前先确认野外并收起大地图，待背景遮罩消退后用新帧核对章节，避免暗化角标造成 OCR 空结果。
    accept_centered_ring 供随后还会实测定位的调用者使用：箭头缺失或微调无效时，
    连续两帧圆环都接近注视点且稳定才接受视野。返回 True 表示使用离屏图标或此独立证据回正，
    不用箭头位置再次否决，调用方仍负责新的小队定位。
    全部输入经过原会话闸门，窗口、预算、取消及战斗异常不在此捕获。
    仅机关触发后可提供 mechanism_position，在两种小队标记均缺失时按新配准的视野执行
    一次中央区域短拖；之后仍须恢复原视觉回正判据，不把投影位置直接当作小队定位。
    """
    session.check()
    session.await_field()
    goto.map_close(session.win)
    session.pause(VERIFY_INTERVAL)
    session.verify_identity(session.await_field())
    evidence = centering_frame(session)
    if mechanism_position is not None and evidence['ground'] is None and evidence['offscreen'] is None:
        recenter_mechanism_edge(session, mechanism_position)
        evidence = centering_frame(session)
    clicks, adjustments, drags = 0, 0, 0
    stalled_adjustments = 0
    searching = False
    while True:
        marker = evidence['offscreen']
        if marker is not None and clicks < 2:
            session.pause(VERIFY_INTERVAL)
            checked = centering_frame(session)
            current = checked['offscreen']
            if (current is not None
                    and np.linalg.norm(np.asarray(current['center']) - marker['center']) <= 12.):
                session.click_client(current['center'])
                clicks += 1
                logger.info(f"小队离屏图标已点击：客户区位置 {current['center']}，第 {clicks}/2 次。")
                return True
            evidence = checked
        if accept_centered_ring and evidence['ground'] is None:
            evidence, ring = verify_centered_compact_ring(session, evidence)
            if ring is not None:
                session.record('recovery_centered_ring_no_arrow', force=True, drags=drags, ring=ring.tolist())
                logger.info('小队箭头缺失，已连续两帧确认紧凑小队圆环在注视点附近。')
                return True
        if not searching:
            if centered(evidence):
                session.pause(VERIFY_INTERVAL)
                checked = centering_frame(session)
                if (centered(checked)
                        and np.linalg.norm(np.asarray(checked['ground']) - evidence['ground'])
                        <= GROUND_STABLE_TOLERANCE):
                    session.record('recovery_centered', force=True, clicks=clicks, drags=drags,
                                   adjustments=adjustments, **checked)
                    logger.info(f'小队居中已确认：图标点击 {clicks} 次，定向拖动 {drags} 次，微调 {adjustments} 次。')
                    return clicks > 0
                evidence = checked
                continue
            ground = evidence['ground']
            if ground is not None:
                offset = np.asarray(goto.VIEWPORT_PIXEL) - ground
                distance = float(np.linalg.norm(offset))
                if distance > CENTER_TOLERANCE:
                    pan_scene(session, offset * min(1., 560. / distance))
                    adjustments += 1
                    evidence = centering_frame(session)
                    updated = evidence['ground']
                    if (updated is not None
                            and distance - np.linalg.norm(goto.VIEWPORT_PIXEL - updated) >= CENTER_MIN_PROGRESS):
                        stalled_adjustments = 0
                    else:
                        stalled_adjustments += 1
                    if stalled_adjustments >= CENTER_STALL_LIMIT or adjustments >= CENTER_ADJUSTMENT_LIMIT:
                        searching = not centered(evidence)
                        if searching and accept_centered_ring:
                            evidence, ring = verify_centered_compact_ring(session, evidence)
                            if ring is not None:
                                session.record('recovery_centered_ring', force=True, adjustments=adjustments,
                                               ring=ring.tolist())
                                logger.info('回正短拖未见效，已连续两帧确认紧凑小队圆环在注视点附近。')
                                return True
                        if searching:
                            logger.info(f'小队回正短拖连续无有效进展 {stalled_adjustments} 次，'
                                        f'累计微调 {adjustments} 次，转入定向搜索。')
                    continue
        if drags == len(RECOVERY_DRAGS):
            break
        ground = evidence['ground']
        drag = np.asarray(RECOVERY_DRAGS[drags])
        drags += 1
        searching = True
        pan_scene(session, drag)
        logger.info(f'小队回正定向拖动：第 {drags}/{len(RECOVERY_DRAGS)} 次，位移 {drag.tolist()} 客户区像素。')
        evidence = centering_frame(session)
    raise ValueError('小队回正未确认：定向拖动及图标点击已达上限，禁止沿用异常前的定位。')
