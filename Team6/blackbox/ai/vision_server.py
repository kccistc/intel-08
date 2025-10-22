import os
import sys
import time
import threading
import json
import math
from queue import Queue

import numpy as np
import cv2
import onnxruntime as ort

#-------------------------------imsi-----------------------------------

import fps_calc
import multiprocessing
import pre_post_process
import core
import demo_manager
import async_api
import visualization
import recorder


# ---- Hailo ----
from hailo_platform import (HEF, Device, VDevice, HailoSchedulingAlgorithm)

# ---- GStreamer ----
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


color_map = [
    (0, 0, 255),   
    (0, 127, 255),   
    (0, 255, 127),   
    (0, 255, 0), 
    (127, 255, 0),   
    (255, 255, 0),   
    (255, 127, 0),   
    (255, 0, 0) ,
    (255,0,127),
    (255,0,255),
    (127,0,255)
]
mapped_class_names = ['car',
                    'truck',
                    'construction_vehicle',
                    'bus',
                    'trailer',
                    'barrier',
                    'motorcycle',
                    'bicycle',
                    'pedestrian',
                    'traffic_cone']
DEBUGMODE = False
# =================== 설정 ===================
# 입력(카메라)
SRC_W, SRC_H = 800, 450
NUM_CAMS = 6
PORT0 = 5000  # udpsrc 시작 포트 (5000~5005)

# 모델 경로
MODELS_DIR = os.path.dirname(__file__)
BACKBONE_HEF    = os.path.join(MODELS_DIR, "petrv2_repvggB0_backbone_pp_800x320.hef")
TRANSFORMER_HEF = os.path.join(MODELS_DIR, "petrv2_repvggB0_transformer_pp_800x320.hef")
POSTPROC_ONNX   = os.path.join(MODELS_DIR, "petrv2_postprocess.onnx")
MATMUL_NPY      = os.path.join(MODELS_DIR, "matmul.npy")
MAP_PATH       = os.path.join(MODELS_DIR, "map.png")
from pathlib import Path
RECORDER_PATH = Path(os.environ.get("BB_OUT_DIR", Path.home() / "blackbox" / "event6"))
RECORDER_PATH.mkdir(parents=True, exist_ok=True)
MAX_QUEUE_SIZE = 3
# BEV 렌더링
BEV_SIZE = 640
BUF_BEV_MAP = BEV_SIZE * 1.5 # 회전 생각해서
XY_RANGE_M = 61.2
LINE_W = 2  # pixel
CARLA_POS_MAX = 210 #pixel
CARLA_POS_MIN = -210 #pixel
MAP_SIZE = 8192  # pixels
MAP_SCALE = (MAP_SIZE - 1) / float(CARLA_POS_MAX - CARLA_POS_MIN)  # 8191/480
BEV_OVERSCAN = 1.2

# 이벤트 비트 설정
EVENT_ACCEL         = 1 << 0
EVENT_BRAKE         = 1 << 1
EVENT_PEDESTRIAN    = 1 << 2
EVENT_TROTTLE       = 1 << 3
EVENT_TRUCK         = 1 << 4
EVENT_MOTORCYCLE    = 1 << 5
EVENT_PUNK          = 1 << 6
EVENT_NONE          = 1 << 7

# ===== Dashboard Composer: 800x450 고정 레이아웃 =====
LCD_W, LCD_H = 800, 450
RIGHT_W = 480           # BEV 목표 가로폭(고정)
LEFT_W  = LCD_W - RIGHT_W  # 320
LEFT_H  = LCD_H

# ================== Map Center/Scale ==================
ORIGIN_PX = MAP_SIZE - 2920  # = MAP_CENTER_X
ORIGIN_PY = MAP_SIZE / 2  # = MAP_CENTER_Y


#상태 전역 변수
global_FRad = 0
global_FFloat_Pos   = (0.0,0.0)
global_SFloat_Pos   = (0.0,0.0)
global_Init_Rad_Flag= False    #최초 2번만 위치정보를 담아서 FRad를 계산해줘야함
global_Init_Rad_Com = False
#================== imsi map ====================
HEADING_OFFSET_DEG = 90.0   # +x(오른쪽)을 '화면 위쪽'으로 돌리기 위한 기본 옵셋
HEADING_EMA = 0.5          # 헤딩 지터 완화(0=안함, 0.2~0.5 추천)
_last_xy = None
_heading_rad = 0.0
_rot_deg_vis = 90.0

def _wrap180(a_deg: float) -> float:
    # [-180, +180)로 래핑
    return (a_deg + 180.0) % 360.0 - 180.0

def _heading_deg_from_xy(prev_xy, curr_xy):
    dx = curr_xy[0] - prev_xy[0]
    dy = curr_xy[1] - prev_xy[1]
    return math.degrees(math.atan2(dy, dx))  # 월드 기준: +x=0°, +y=+90°

def update_rot_deg_from_gps(curr_xy, move_eps_m: float = 0.15):
    """
    GPS로부터 목표 회전각(target_rot)을 만들고,
    현재 표시각(_rot_deg_vis)을 shortest-arc로 그쪽으로 EMA 보정.

    반환: 새 _rot_deg_vis (도)
    """
    global _last_xy, _rot_deg_vis

    if _last_xy is None:
        _last_xy = curr_xy
        return _rot_deg_vis

    dx = curr_xy[0] - _last_xy[0]
    dy = curr_xy[1] - _last_xy[1]
    if dx*dx + dy*dy < move_eps_m*move_eps_m:
        # 거의 안 움직이면 방향 갱신 안 함 (노이즈 회피)
        return _rot_deg_vis

    heading_deg = _heading_deg_from_xy(_last_xy, curr_xy)   # 월드 헤딩
    target_rot  = 90.0 - heading_deg                        # 화면 회전각(도): +x→위
    delta       = _wrap180(target_rot - _rot_deg_vis)        # ✅ shortest-arc
    _rot_deg_vis += HEADING_EMA * delta                      # EMA 보정

    _last_xy = curr_xy
    return _rot_deg_vis

def _draw_rot_square(img, center_xy, size_px, angle_deg, color=(0,255,255), thickness=2):
    """중심/한변 size_px 회전 정사각형을 img에 그림 (각도: 시계+ 기준)"""
    cx, cy = center_xy
    c = size_px / 2.0
    pts = np.array([[-c,-c],[+c,-c],[+c,+c],[-c,+c]], np.float32)
    rad = math.radians(angle_deg)
    co, si = math.cos(rad), math.sin(rad)
    R = np.array([[co,-si],[si,co]], np.float32)
    rot = (R @ pts.T).T
    rot[:,0] += cx; rot[:,1] += cy
    pts_i = rot.astype(np.int32)
    cv2.polylines(img, [pts_i], True, color, thickness)

def render_fullmap_8192(map_image, x, y, yaw_rad, xy_range_m=XY_RANGE_M,
                        out_size=(480,480), path_tail=None):
    """
    - 8192 원본에 현재 위치/방향, BEV 크롭 영역을 그려 넣고
    - 최종만 out_size로 리사이즈해서 반환
    yaw_rad: 라디안(헤딩). BEV 회전과 동일 부호로 사용
    """
    # 1) 원본 복사(8192 메모리 크므로 반드시 copy 후 작업)
    canvas = map_image.copy()

    # 2) 좌표 변환
    px, py = world_to_pixel(x, y)

    # 3) 궤적(선택): 최근 포인트들을 원본 좌표계로 그려줌
    if path_tail and len(path_tail) > 1:
        for i in range(1, len(path_tail)):
            p0 = world_to_pixel(*path_tail[i-1])
            p1 = world_to_pixel(*path_tail[i])
            cv2.line(canvas, p0, p1, (0,200,0), 1)

    # 4) 현재 위치(빨간 점)
    cv2.circle(canvas, (px, py), 4, (0,0,255), -1)

    # 5) 진행방향 화살표(길이 10m를 픽셀로 환산)
    Lm = 10.0
    Lpx = int(round(Lm * MAP_SCALE))
    # 지도는 y-down이므로 y방향 부호 주의
    dx = int(round(Lpx * math.cos(yaw_rad)))
    dy = int(round(-Lpx * math.sin(yaw_rad)))
    cv2.arrowedLine(canvas, (px, py), (px + dx, py + dy), (0,0,255), 2, tipLength=0.2)

    # 6) BEV 크롭 영역(±xy_range_m): 한 변 픽셀 길이 = 2*range(m)*MAP_SCALE
    crop_px = int(round(2 * xy_range_m * MAP_SCALE))
    angle_deg = -math.degrees(float(yaw_rad))  # BEV와 부호 일관
    _draw_rot_square(canvas, (px, py), 480, angle_deg, (0,255,255), 2)

    # 7) 정보 텍스트(좌표/각도)
    text = f"x={x:.1f} y={y:.1f} yaw={math.degrees(yaw_rad):.1f}°"
    cv2.putText(canvas, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    # 8) 최종만 리사이즈해서 창에 표시할 프레임 리턴
    if out_size is not None:
        canvas = cv2.resize(canvas, out_size, interpolation=cv2.INTER_AREA)
    return canvas

def world_to_pixel(x, y):
    """CARLA y-up → 이미지 y-down 보정 포함, 8192 범위로 클램프"""
    x = -x
    px = ORIGIN_PX + x * MAP_SCALE
    py = ORIGIN_PY - y * MAP_SCALE
    px = int(round(max(0, min(px, MAP_SIZE - 1))))
    py = int(round(max(0, min(py, MAP_SIZE - 1))))
    return px, py

# imsi 지금 Carla에서는 좌수계라고 해서 y축 x축이 조금 다를수 있음 확인해보면서 각도 잘 되는지 확인피료앟ㅁ
def init_global_FRad(global_Float_Pos, global_SFloat_Pos):
    global_FRad = 0
    delta_x = global_SFloat_Pos[0] - global_Float_Pos[0]
    delta_y = global_SFloat_Pos[1] - global_Float_Pos[1]
    if delta_x == 0 and delta_y == 0:
        global_FRad = 0
    else:
        global_FRad = math.atan2(delta_y, delta_x)

    return global_FRad


# 이벤트 플래그를 문자열로 변환시킬거임
def recoder_event(event_flags):

    parts = []
    if event_flags & EVENT_ACCEL:       parts.append("accel")
    if event_flags & EVENT_BRAKE:       parts.append("brake")
    if event_flags & EVENT_PEDESTRIAN:  parts.append("pedestrian")
    if event_flags & EVENT_TROTTLE:     parts.append("throttle")
    if event_flags & EVENT_TRUCK:       parts.append("truck")
    if event_flags & EVENT_MOTORCYCLE:  parts.append("motorcycle")
    if event_flags & EVENT_PUNK:        parts.append("punk")

    return "_".join(parts)

def log(msg: str):
    print(f"[Py LOG] {msg}", file=sys.stderr, flush=True)

# =================== 유틸 (qparam/양자화/BEV) ===================
import queue as _queue

def _np_to_py(obj):
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_to_py(v) for v in obj]
    return obj
def q1(x): return float(np.around(x, 1))

def _resize_keep_ar_by_width(img, target_w):
    """가로를 target_w로 맞추고, 세로는 비율 유지."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((1, target_w, 3), np.uint8)
    new_h = int(round(h * (target_w / float(w))))
    return cv2.resize(img, (target_w, new_h))

def _center_paste(dst, tile, x, y, w, h):
    """dst[y:y+h, x:x+w] 영역 가운데에 tile을 레터박스 방식으로 붙임(넘치면 잘림)."""
    th, tw = tile.shape[:2]
    # 타겟 영역 내 중앙 정렬
    ox = x + max(0, (w - tw) // 2)
    oy = y + max(0, (h - th) // 2)
    # 클리핑
    xs = max(x, ox); xe = min(x + w, ox + tw)
    ys = max(y, oy); ye = min(y + h, oy + th)
    if xs < xe and ys < ye:
        dst[ys:ye, xs:xe] = tile[(ys-oy):(ys-oy)+(ye-ys), (xs-ox):(xs-ox)+(xe-xs)]

# 🌟🌟🌟 수정된 Dashboard 생성 함수 🌟🌟🌟
def compose_dashboard_800x450_mosaic_left_bev_right(mosaic, bev_480x480):
    """
    최종 출력: 800x450 BGR
      - 왼쪽(400x450): Mosaic 이미지를 400 가로폭에 맞춰 비율 유지 리사이즈 후, 패널 중앙에 배치.
      - 오른쪽(400x450): BEV 480x480을 400x400으로 리사이즈 후, 패널 중앙에 배치.
    """
    # ===== 레이아웃 설정 =====
    LCD_W, LCD_H = 800, 450
    LEFT_W = 400
    RIGHT_W = LCD_W - LEFT_W # 400

    # 1. 최종 캔버스 생성
    canvas = np.zeros((LCD_H, LCD_W, 3), np.uint8)

    # 2. 오른쪽 BEV 패널 처리
    # 480x480 BEV를 400x400으로 리사이즈
    if bev_480x480 is None:
        bev_400 = np.zeros((400, 400, 3), np.uint8)
    else:
        bev_400 = cv2.resize(bev_480x480, (400, 400), interpolation=cv2.INTER_AREA)

    # 🌟 해결 1: 400x450 오른쪽 패널의 중앙에 400x400 BEV를 붙여넣기 (크롭 대신)
    _center_paste(canvas, bev_400, x=LEFT_W, y=0, w=RIGHT_W, h=LCD_H)

    # 3. 왼쪽 Mosaic 패널 처리
    if mosaic is None:
        mosaic = np.zeros((10, 10, 3), np.uint8)

    # 🌟 해결 2: Mosaic 이미지의 가로폭을 LEFT_W(400)로 맞춤 (비율 유지)
    mosaic_resized = _resize_keep_ar_by_width(mosaic, LEFT_W)

    # 400x450 왼쪽 패널의 중앙에 리사이즈된 Mosaic 붙여넣기
    _center_paste(canvas, mosaic_resized, x=0, y=0, w=LEFT_W, h=LCD_H)

    # 4. 테두리 그리기
    cv2.rectangle(canvas, (0,0), (LEFT_W-1, LCD_H-1), (60,60,60), 1)
    cv2.rectangle(canvas, (LEFT_W,0), (LCD_W-1, LCD_H-1), (60,60,60), 1)

    return canvas

def _build_json_msg(dets_dict, meta=None, score_thresh=0.3):

    objs = []
    try:
        if isinstance(dets_dict, dict) and dets_dict.get('pts_bbox'):
            item0 = dets_dict['pts_bbox'][0] or {}
            boxes  = np.asarray(item0.get("boxes_3d", []))
            scores = np.asarray(item0.get("scores_3d", []))
            labels = np.asarray(item0.get("labels_3d", []))

            if boxes.ndim == 1:  # (7,) 같은 단일 케이스 방어
                boxes = boxes.reshape(1, -1)

            N = boxes.shape[0] if boxes.size else 0
            for i in range(N):
                if scores[i] < score_thresh:
                    continue

                b = boxes[i]
                l = labels[i]
                obj = {
                    "label": _np_to_py(l),
                    "x":     q1(b[0]),  # 0번
                    "y":     q1(b[1]),  # 1번
                    "ax": q1(b[7]),  # 7번
                    "ay":  q1(b[8]),  # 8번
                }
                #log(f"[DEBUG] box {i}: {obj}")  # 디버그 로그
                if i < len(scores): obj["score"] = _np_to_py(scores[i])
                if i < len(labels): obj["label"] = _np_to_py(labels[i])
                objs.append(obj)
    except Exception as e:
        # 디코드가 비정상이면 빈 배열 + 상태만 리턴
        pass

    return objs          # ← 요청한 배열 키



def json_sender_proc(det_q):
    import sys, json
    import pre_post_process as pp
    while True:
        try:
            item = det_q.get(timeout=1)
        except _queue.Empty:
            continue

        meta = None
        if isinstance(item, tuple) and len(item) == 2:
            pp_output, meta = item
            try:
                dets_dict = pp.decode(pp_output)
            except Exception:
                dets_dict = {}
        else:
            dets_dict = item if isinstance(item, dict) else {}

        obj = {
            "objects": _build_json_msg(dets_dict, meta=meta),
        }

        print(json.dumps(obj, ensure_ascii=False))
        sys.stdout.flush()

def parse_command(line: str):
    """
    'analyze {json}' 형태에서 payload만 파싱 (없어도 동작은 하게 관대하게 처리)
    """
    line = line.strip()
    if not line:
        return None, None

    if line.startswith("analyze"):

        payload = None
        # 'analyze ' 뒤에 JSON이 붙어 있을 수도 있음
        if len(line) > len("analyze"):
            rest = line[len("analyze"):].strip()
            if rest:
                try:
                    payload = json.loads(rest)
                except Exception:
                    payload = None
        return "analyze", payload


    elif line.startswith("draw"):

        payload = None
        # 'analyze ' 뒤에 JSON이 붙어 있을 수도 있음
        if len(line) > len("draw"):
            rest = line[len("draw"):].strip()
            if rest:
                try:
                    payload = json.loads(rest)
                except Exception:
                    payload = None
        return "draw", payload



# === Add to vision_server.py (상단 유틸 근처) ===

# =================== map ====================


def render_bev_frame(map_image, in_queue, payload, xy_range=XY_RANGE_M, size=640):
    import pre_post_process as pp
    last_dets = None

    # --- 디텍션 최신값(있으면) 가져오기 ---
    try:
        item = in_queue.get(timeout=0.5)
        if isinstance(item, tuple) and len(item) == 2:
            pp_output, _meta = item
            last_dets = pp.decode(pp_output)
        else:
            last_dets = item
    except _queue.Empty:
        last_dets = None

    # --- payload 해석 ---
    payload_draw, analyze_payload = payload
    gps = analyze_payload['gps']      # (x, y)
    x, y = gps

    # === 핵심: 좌표로 헤딩 갱신 → 회전각(도) 산출 ===
    angle_deg = update_rot_deg_from_gps((x, y))

    # === 회전 & 크롭 ===
    cx, cy = world_to_pixel(x, y)
    # xy_range(미터) → 픽셀 변환하고, 오버스캔 적용
    pre_crop_px = int(round(2 * xy_range * MAP_SCALE * BEV_OVERSCAN))
    # 너무 작거나 너무 크면 가드
    pre_crop_px = max(size, min(pre_crop_px, MAP_SIZE - 2))
    bev_big = rotate_and_crop_constant(
        map_image,
        angle_deg=-angle_deg,
        crop_w=pre_crop_px, crop_h=pre_crop_px,
        center=(cx, cy),
        border_value=(255,255,255)  # 흰색
    )
    bev = cv2.resize(bev_big, (size, size), interpolation=cv2.INTER_AREA)

    ox = oy = size // 2
    m = 5  # 반쪽 길이(px) → 네모는 (2m x 2m)
    cv2.rectangle(bev, (ox - m, oy - m*2), (ox + m, oy + m*2), (0, 255, 0), 2)
    # === 박스 그리기 ===
    if last_dets:
        bev, _ = draw_bev_boxes_on(bev, last_dets, score_thresh=0.3, xy_range=xy_range)

    return bev, payload_draw


def compose_dashboard_800x450_two_imgs_left_bev_right(img_top, img_bottom, bev_480x480):
    """
    최종 출력: 800x450 BGR
      - 왼쪽(320x450): 위/아래 이미지 2장 (가로 320 맞춤, 세로는 비율 유지, 남는 공간은 여백)
      - 오른쪽(480x450): BEV 480x480을 세로 중앙 30px 크롭(위15, 아래15) → 480x450
    """
    canvas = np.zeros((LCD_H, LCD_W, 3), np.uint8)

    # ---- Right: BEV 480x450 (세로 중앙 크롭) ----
    bev = bev_480x480
    if bev.shape[0] != 480 or bev.shape[1] != 480:
        bev = cv2.resize(bev, (480, 480))  # 안전장치

    top_crop = 15
    bev_cropped = bev[top_crop:top_crop+LCD_H, 0:RIGHT_W]  # (450, 480, 3)
    canvas[0:LCD_H, LEFT_W:LEFT_W+RIGHT_W] = bev_cropped

    # ---- Left: 1열 2행 (320x450 패널) ----
    left_panel = np.zeros((LEFT_H, LEFT_W, 3), np.uint8)

    # 각 타일의 배치 높이(수직 2분할, 중간 8px 간격 권장)
    GAP = 8
    slot_h = (LEFT_H - GAP) // 2  # 위/아래 슬롯 높이

    # 위 타일
    if img_top is None:
        img_top = np.zeros((10, 10, 3), np.uint8)
    top_resized = _resize_keep_ar_by_width(img_top, LEFT_W)
    _center_paste(left_panel, top_resized, x=0, y=0, w=LEFT_W, h=slot_h)

    # 아래 타일
    if img_bottom is None:
        img_bottom = np.zeros((10, 10, 3), np.uint8)
    bot_resized = _resize_keep_ar_by_width(img_bottom, LEFT_W)
    _center_paste(left_panel, bot_resized, x=0, y=slot_h + GAP, w=LEFT_W, h=slot_h)

    # 합성
    canvas[0:LEFT_H, 0:LEFT_W] = left_panel
    # 테두리(선택)
    cv2.rectangle(canvas, (0,0), (LEFT_W-1, LEFT_H-1), (60,60,60), 1)
    cv2.rectangle(canvas, (LEFT_W,0), (LCD_W-1, LCD_H-1), (60,60,60), 1)

    return canvas





import math
import numpy as np
import cv2

def _rot_rect_corners(cx, cy, w, l, yaw):
    # 중심(cx,cy), 폭(w), 길이(l), 라디안 yaw(차량 진행방향이 +x 기준)
    # 4코너 (앞-왼, 앞-오, 뒤-오, 뒤-왼) 시계방향
    hw, hl = w/2.0, l/2.0
    corners = np.array([
        [ +hl, +hw],  # front-right (x forward, y left 기준이면 순서 맞춰도 무방)
        [ +hl, -hw],
        [ -hl, -hw],
        [ -hl, +hw],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    rot = (R @ corners.T).T
    rot[:,0] += cx
    rot[:,1] += cy
    return rot  # (4,2) in meters (BEV 좌표계)

def _meters_to_pixels(xy, origin_px, scale):
    # origin_px = (ox, oy), scale = px per meter
    xy_px = xy.copy()
    xy_px[:,0] = origin_px[0] + xy[:,0]*scale
    xy_px[:,1] = origin_px[1] - xy[:,1]*scale  # y축 뒤집기(위가 +)
    return xy_px.astype(np.int32)

def draw_bev_boxes_on(canvas, dets, score_thresh=0.30, xy_range=61.2):
    # dets: pre_post_process.decode() 결과(dict)
    if not dets or not dets.get('pts_bbox'):
        return canvas, (0, 3) # 기본 pos 반환
    item = dets['pts_bbox'][0]
    boxes3d = np.asarray(item.get('boxes_3d', []))  # (N, >=7) = [cx, cy, cz, w, l, h, yaw, ...]
    scores  = np.asarray(item.get('scores_3d', []))
    labels  = np.asarray(item.get('labels_3d', []))

    if boxes3d.size == 0:
        return canvas, (0, 3) # 기본 pos 반환

    H, W = canvas.shape[:2]
    ox = oy = W//2
    scale = W/(2*xy_range)

    region_counts = [0] * 6

    for i in range(len(boxes3d)):
        s = float(scores[i]) if i < len(scores) else 1.0
        if labels[i] == 4 or labels[i] == 5 or labels[i] == 9 :
            continue
        if s < score_thresh:
            continue
        cx, cy, cz, l, w, h, yaw = boxes3d[i, :7]

        # 범위 밖이면 스킵
        if abs(cx) > xy_range or abs(cy) > xy_range:
            continue

        # 영역 카운팅 로직
        angle_deg = np.degrees(np.arctan2(cy, cx))
        region_index = -1
        if cx >= 0: # 전방
            if 30 <= angle_deg < 90:    region_index = 2
            elif -30 <= angle_deg < 30: region_index = 1
            elif -90 <= angle_deg < -30: region_index = 0
        else: # 후방
            if 90 <= angle_deg < 150:    region_index = 3
            elif 150 <= angle_deg or angle_deg < -150: region_index = 4
            elif -150 <= angle_deg < -90: region_index = 5
        if region_index != -1:
            region_counts[region_index] += 1

        # 코너 4점(m) → 픽셀
        corners_m = _rot_rect_corners(cx, cy, w, l, -yaw)
        corners_px = _meters_to_pixels(corners_m, (ox, oy), scale)

        # 박스
        cv2.polylines(canvas, [corners_px], isClosed=True, color=color_map[labels[i]], thickness=2)

        # 진행방향(앞쪽 에지의 중점)
        front_mid_m = (corners_m[0] + corners_m[1]) / 2.0
        center_px = _meters_to_pixels(np.array([[cx,cy]]), (ox,oy), scale)[0]
        front_px  = _meters_to_pixels(np.array([front_mid_m]), (ox,oy), scale)[0]
        cv2.line(canvas, center_px, front_px, (0,0,255), 2)

        # 라벨/점수
        cv2.putText(canvas, f"{mapped_class_names[labels[i]]}", front_px, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,20,20), 1, cv2.LINE_AA)#아오

    # 가장 많이 검출된 영역 계산
    front_counts = region_counts[0:3]
    rear_counts = region_counts[3:6]
    max_front_region_index = np.argmax(front_counts)
    max_rear_region_index = np.argmax(rear_counts) + 3
    pos = (max_front_region_index, max_rear_region_index)

    return canvas, pos

# === 3x2 모자이크 유틸 ===
def draw_cam_index(img, idx):
    cv2.putText(img, f"Cam {idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, f"Cam {idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return img

def make_mosaic_grid(img_list, rows=2, cols=3, tile_wh=(400,160), pad=4, order=None, draw_index=True):
    """
    img_list: [H,W,3] BGR 이미지 리스트 (최대 rows*cols개)
    tile_wh: 한 타일 크기 (w,h)
    pad: 타일 간 여백(px)
    order: [이미지 인덱스 배치 순서], 예) [0,1,2,3,4,5]
    """
    if order is None:
        order = list(range(len(img_list)))
    # 캔버스 크기 계산
    tw, th = tile_wh
    grid_w = cols * tw + (cols + 1) * pad
    grid_h = rows * th + (rows + 1) * pad
    canvas = np.zeros((grid_h, grid_w, 3), np.uint8)

    # 배치
    for k in range(rows * cols):
        r = k // cols
        c = k % cols
        x0 = pad + c * (tw + pad)
        y0 = pad + r * (th + pad)

        if k < len(order) and order[k] < len(img_list) and img_list[order[k]] is not None:
            tile = cv2.resize(img_list[order[k]], (tw, th))
            if draw_index:
                tile = draw_cam_index(tile, order[k])
        else:
            tile = np.zeros((th, tw, 3), np.uint8)

        canvas[y0:y0+th, x0:x0+tw] = tile

    # 테두리
    cv2.rectangle(canvas, (0,0), (grid_w-1, grid_h-1), (40,40,40), 1)
    return canvas



def rotate_and_crop_constant(img, angle_deg, crop_w, crop_h,
                             center=None,  # (px, py) 회전/크롭 중심. None이면 이미지 중앙
                             interpolation=cv2.INTER_LINEAR,
                             border_value=(255,255,255)):

    import math, numpy as np, cv2
    H, W = img.shape[:2]
    cx, cy = center

    # 1) 회전에 충분한 "사전 크롭" 크기(r): 대각선 절반 + 여유
    r = int(math.ceil(0.5 * math.hypot(crop_w, crop_h))) + 4  # 640 → r≈454

    x0 = max(0, int(cx - r)); y0 = max(0, int(cy - r))
    x1 = min(W, int(cx + r)); y1 = min(H, int(cy + r))

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        # 밖으로 나갔을 때 빈 캔버스 반환
        return np.full((crop_h, crop_w, img.shape[2] if img.ndim==3 else 1),
                       border_value, dtype=img.dtype)

    # 2) ROI 기준 중심 좌표
    rcx = cx - x0; rcy = cy - y0

    # 3) ROI만 회전
    M = cv2.getRotationMatrix2D((rcx, rcy), angle_deg, 1.0)
    rotated = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                             flags=interpolation,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)

    # 4) 회전한 ROI에서 중앙 crop_w×crop_h만 떼기
    x0c = int(round(rcx - crop_w/2)); y0c = int(round(rcy - crop_h/2))
    x1c = x0c + crop_w; y1c = y0c + crop_h

    # 겹치는 영역만 복사 (경계 넘어가면 0으로 채움)
    out = np.full((crop_h, crop_w, rotated.shape[2] if rotated.ndim==3 else 1),
                  border_value, dtype=rotated.dtype)
    sx0 = max(0, x0c); sy0 = max(0, y0c)
    sx1 = min(rotated.shape[1], x1c); sy1 = min(rotated.shape[0], y1c)
    if sx0 < sx1 and sy0 < sy1:
        dx0 = sx0 - x0c; dy0 = sy0 - y0c
        out[dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = rotated[sy0:sy1, sx0:sx1]

    return out
# =================== GStreamer 수신 ===================
class GstVideoReceiver:
    def __init__(self, port: int):
        self.port = port
        self.pipeline = None
        self.appsink = None
        self.latest_frame = None
        self._stop = False
        self._th = None

    def init_pipeline(self):
        # appsink를 BGR로 맞춤
        pipeline_str = (
            f"udpsrc port={self.port} ! "
            "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=60 ! "
            "rtph264depay ! h264parse config-interval=-1 ! "
            "avdec_h264 max-threads=0 ! videoconvert ! "
            "queue max-size-buffers=5 ! "  # 추가
            "video/x-raw,format=BGR ! appsink name=sink drop=true max-buffers=1 sync=false"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)

    def _pull(self):
        # 일부 환경은 try_pull_sample 바인딩이 없어서 action-signal 사용
        sample = self.appsink.emit("try-pull-sample", 50 * Gst.MSECOND)
        if not sample:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = int(s.get_value("width"))
        h = int(s.get_value("height"))
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            return arr.reshape((h, w, 3))
        finally:
            buf.unmap(mapinfo)

    def _loop(self):
        import time
        idle_sleep = 0.005   # 5ms만 쉬어도 효과 큼
        while not self._stop:
            f = self._pull()
            if f is None:
                time.sleep(idle_sleep)   # ← 이 한 줄이 코어 100%를 크게 내림
                continue
            self.latest_frame = f

    def start(self):
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop = True
        if self._th:
            self._th.join()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

det_for_json_q = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
det_for_bev_q  = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

STOP = ("__STOP__", None)

def fanout_proc(src_q, dst_qs):
    import queue as _q
    while True:
        try:
            item = src_q.get(timeout=0.5)
        except _q.Empty:
            continue
        # 각 목적지 큐로 non-blocking 전송 (풀이면 드롭)
        for q in dst_qs:
            try: q.put_nowait(item)
            except: pass
# =================== 메인 ===================
def main():
    log("Simple BEV server starting")

    # GStreamer 시작
    Gst.init(None)
    receivers = [GstVideoReceiver(PORT0 + i) for i in range(NUM_CAMS)]
    for r in receivers:
        r.init_pipeline()
        r.start()

    # Map 데이터 로드
    map_image = cv2.imread(MAP_PATH)
    if map_image is None:
        log(f"Error: Map image not found at {MAP_PATH}")
        map_image = np.zeros((MAP_SIZE, MAP_SIZE, 3), np.uint8) # Fallback to black image


    # Recorder 초기화
    rec = recorder.TimeWindowEventRecorder6(
         out_dir=str(RECORDER_PATH),
         size=(800, 450),
         pre_secs=5.0, post_secs=5.0,
         retention_secs=15.0,
         save_as="mp4",
         target_fps=5.0,
         exact_count=False
    )


    fps_calculator = fps_calc.FPSCalc(2)
    queues = []
    bb_tranformer_meta_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    transformer_pp_meta_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    bb_tranformer_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    transformer_pp_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    pp_3dnms_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)


    # Hailo VDevice
    manager = multiprocessing.Manager()
    demo_mng = demo_manager.DemoManager(manager)
    device_ids = Device.scan()
    if not device_ids:
        raise RuntimeError("Hailo 디바이스가 없습니다. (모듈/권한 확인: lsmod | grep -i hailo, /dev/hailo0 권한)")
    log(f"[HAILO] found devices: {device_ids}")


    params = VDevice.create_params()
    if hasattr(params, "device_ids"):
        params.device_ids = device_ids
    threads = []
    processes = []
    with VDevice(params) as target:
        log("[HAILO] VDevice ready")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as target:

        camera_in_q = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

        threads.append(threading.Thread(target=core.backbone_from_cam, args=(target, camera_in_q, BACKBONE_HEF, bb_tranformer_queue,
                                    bb_tranformer_meta_queue, demo_mng, True)))

        threads.append(threading.Thread(target=core.transformer, args=(target, TRANSFORMER_HEF, MATMUL_NPY, bb_tranformer_queue, bb_tranformer_meta_queue, transformer_pp_queue, transformer_pp_meta_queue,
                                                                       demo_mng,2.15,-5.3)))

        processes.append(multiprocessing.Process(target=pre_post_process.post_proc,
                                                    args=(transformer_pp_queue, transformer_pp_meta_queue,
                                                    pp_3dnms_queue, POSTPROC_ONNX, demo_mng)))

        log("multi process starting...")
        for t in threads: t.start()
        for p in processes: p.start()

        # fanout
        fan = multiprocessing.Process(target=fanout_proc, args=(pp_3dnms_queue, [det_for_json_q, det_for_bev_q]))
        fan.daemon = False
        fan.start()

        # 각 소비자에는 자기 큐만 전달
        json_out = multiprocessing.Process(target=json_sender_proc, args=(det_for_json_q,))
        json_out.daemon = False
        json_out.start()

        token  = 0
        recodCMD = 0

        log("init done")
        WIN = "Dashboard"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        if not DEBUGMODE: #디버그 안할땐 풀스크린 하면 안됨
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(WIN, 0, 0)

        try :
            while True:

                line = sys.stdin.readline()
                if not line:
                    log("stdin closed. exiting.")
                    break

                cmd, anlalyze_paylaod = parse_command(line)
                if cmd != "analyze":
                    log(f"ignored line in analyze: {line.strip()}")
                    continue

                images_record = []
                if cmd == "analyze":
                    # 1) 6캠 프레임 수집
                    images_after_pre = []
                    for i in range(NUM_CAMS):
                        f = receivers[i].latest_frame
                        if f is None:
                            f = np.zeros((SRC_H, SRC_W, 3), np.uint8)
                        img = cv2.resize(f, (800, 450))
                        images_record.append(img.copy())
                        x, y, width, height = 0, 130, 800, 450
                        img = img[y:height, x:x + width]
                        images_after_pre.append(img)

                    frames_np = np.asarray(images_after_pre, dtype=np.uint8)
                    token += 1

                    try:
                        camera_in_q.put((frames_np, {"token": token}), block=False)
                    except _queue.Full:
                        _ = camera_in_q.get()
                        camera_in_q.put((frames_np, {"token": token}), block=False)
                        log("[Main] WARN: camera_in_q full, dropping frame")

                    log("wating draw cmd...")
                    line = sys.stdin.readline()
                    cmd, payload = parse_command(line)

                    if cmd != "draw":
                        log(f"ignored line in draw: {line.strip()}")
                        print("done", flush=True)
                        continue

                    if cmd == "draw":
                        cam_order = [2, 0, 1, 5, 3, 4]

                        # 🌟🌟🌟 수정된 Mosaic 생성 파라미터 🌟🌟🌟
                        mosaic = make_mosaic_grid(
                            images_record, # 크롭 전 원본(800x450) 사용
                            rows=3, cols=2,
                            tile_wh=(190, 107), # 400px 패널에 맞춘 크기
                            pad=6,
                            order=cam_order,
                            draw_index=True
                        )

                        if DEBUGMODE:
                            # 🌟 수정된 창 이름
                            cv2.imshow("Cams 3x2", mosaic)
                            cv2.waitKey(1)

                        bev_480, payload_draw = render_bev_frame(
                            map_image, det_for_bev_q, (payload, anlalyze_paylaod),
                            xy_range=XY_RANGE_M, size=480
                        )
                        
                        _, pos = draw_bev_boxes_on(bev_480, det_for_bev_q.get() if not det_for_bev_q.empty() else None)
                        
                        img_top    = images_record[0] if len(images_record) > pos[0] else None
                        img_bottom = images_record[3] if len(images_record) > pos[1] else None

                        # Dashboard for display
                        dashboard_display = compose_dashboard_800x450_two_imgs_left_bev_right(img_top, img_bottom, bev_480)

                        H, W = bev_480.shape[:2]
                        txt1 = f"tires {payload['tires'][0]:.1f}, {payload['tires'][1]:.1f}, {payload['tires'][2]:.1f}, {payload['tires'][3]:.1f} event : {payload['value']}"
                        txt2 = f"speed : {payload['speed']:.1f} km/h brake :{payload_draw['brake_state']}%  throttle : {payload_draw['throttle']} % rpm : {payload_draw['rpm']}"

                        # Display Dashboard에 텍스트 그리기
                        cv2.putText(dashboard_display, txt1, (20, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(dashboard_display, txt1, (20, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(dashboard_display, txt2, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(dashboard_display, txt2, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
                        cv2.imshow(WIN, dashboard_display)

                        # Dashboard for recording
                        record_dashboard = compose_dashboard_800x450_mosaic_left_bev_right(mosaic, bev_480)
                        # 🌟 cam_id 추가
                        rec.push_single(record_dashboard, cam_id=0)

                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            break
                        elif key == ord('f'):
                            fs = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if fs == 1.0 else cv2.WINDOW_FULLSCREEN)

                        event = payload['value']
                        SPECIAL_LINE_PATH = "./special_event_log.txt"
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")

                        if ((event & EVENT_NONE) == 0x00) and ((event & 0x3F) != 0x00):
                            log(f"event : {event}")
                            trigger_str = recoder_event(event)
                            rec.trigger(str(trigger_str))
                            with open(SPECIAL_LINE_PATH, "a", encoding="utf-8") as f:
                                f.write(f"[{ts}] {trigger_str} : event value : {event}\n")
                        log("end draw ...")


                print("done", flush=True)
            print("exit", flush=True)
        except KeyboardInterrupt:
            demo_mng.set_terminate()

        finally:
            for q in (pp_3dnms_queue, det_for_json_q, det_for_bev_q):
                try: q.put_nowait(STOP)
                except: pass

            for t in threads: t.join(timeout=1)
            for p in processes: p.join(timeout=2)

            for r in receivers: r.stop()

            cv2.destroyAllWindows()

    log("Done.")


if __name__ == "__main__":
    main()
