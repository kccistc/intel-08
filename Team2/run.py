#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, warnings, threading, queue, subprocess
from datetime import datetime, timedelta

# iotdemo 경로 우선 추가 (사용자 제공 위치)
try:
    sys.path.append(os.path.expanduser('./iotdemo'))
except Exception:
    pass

# ===== Optional deps (GUI/DB/IO) =====
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import pymysql
except Exception:
    pymysql = None
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    Image = None; ImageTk = None; PIL_OK = False
try:
    import cv2
except Exception:
    cv2 = None
try:
    import serial
except Exception:
    serial = None

# (사용자 환경의 iotdemo.* 가 있을 수도/없을 수도 있으므로 try)
try:
    from iotdemo import FactoryController, Inputs, Outputs, PyDuino, PyFt232
except Exception:
    FactoryController = Inputs = Outputs = PyDuino = PyFt232 = None

# ====== ONNX/NumPy ======
import numpy as np
import onnxruntime as ort
from math import pi

# ---------- UI colors ----------
PINK  = (203, 80, 165)   # bbox
WHITE = (255,255,255)

# =========================
# Detection/Seg utilities
# =========================
def letterbox(im, size, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(size/h, size/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=im.dtype)
    top  = (size-nh)//2; left = (size-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def _rounded_rect(img, pt1, pt2, color, thickness=2, r=8):
    x1,y1 = pt1; x2,y2 = pt2
    if r<=0:
        cv2.rectangle(img, pt1, pt2, color, thickness); return
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,thickness)
    cv2.ellipse(img,(x1+r,y2-r),(r,r),90 ,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),0  ,0,90,color,thickness)

def _label_badge(img, x, y, text, bg, fg=WHITE):
    font = cv2.FONT_HERSHEY_SIMPLEX; scale=0.55; thick=2; pad=6; r=6
    (tw,th), _ = cv2.getTextSize(text, font, scale, thick)
    bx1,by1 = x, max(0, y - th - 2*pad - 6)
    bx2,by2 = bx1 + tw + 2*pad + 2*r, by1 + th + 2*pad
    cv2.rectangle(img,(bx1,by1),(bx2,by2),bg,-1)
    _rounded_rect(img,(bx1,by1),(bx2,by2),bg,2,r)
    cv2.putText(img, text, (bx1+pad+r//2, by2-pad), font, scale, fg, thick, cv2.LINE_AA)

# [PATCH] 좌측 상단에 멀티라인 텍스트를 그려주는 헬퍼
def put_lines_top_left(img, lines, colors=None, x=12, y0=22, dy=20):
    """lines: ['text1','text2',...]  colors: [(b,g,r), ...] 또는 None"""
    font = cv2.FONT_HERSHEY_SIMPLEX; scale=0.55; thick=2
    for i, text in enumerate(lines):
        y = y0 + i * dy
        color = (colors[i] if (colors and i < len(colors)) else (255,255,255))
        cv2.putText(img, str(text), (x, y), font, scale, color, thick, cv2.LINE_AA)

def draw_box(frame, x1,y1,x2,y2, score, cls, labels, color=PINK):
    # [PATCH] 박스만 그리고, 텍스트 라벨은 표시하지 않음
    _rounded_rect(frame, (x1,y1), (x2,y2), color, 2, r=8)


def draw_seg_on_roi(roi_bgr, mask_uint8, labels, *, sticker_cid=1, stain_cid=2):
    h,w = mask_uint8.shape
    out = roi_bgr.copy()
    class_plan = [
        (sticker_cid, {"fill":(0,255,255),  "alpha":0.45, "edge":(60,220,60),   "badge":(40,180,40),  "fallback":"sticker"}),
        (stain_cid,   {"fill":(255,120,0),  "alpha":0.45, "edge":(255,210,120), "badge":(200,160,80), "fallback":"stain"}),
    ]
    for cid, style in class_plan:
        if cid < 0:
            continue
        m = (mask_uint8 == cid).astype(np.uint8)
        if m.sum() == 0:
            continue
        overlay = out.copy()
        overlay[m>0] = style["fill"]
        cv2.addWeighted(overlay, style["alpha"], out, 1-style["alpha"], 0, out)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, style["edge"], 2)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                name = labels[cid] if cid < len(labels) else style["fallback"]
                font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.5; thick=1; pad=4
                (tw,th), _ = cv2.getTextSize(name, font, scale, thick)
                bx1,by1 = max(0, cx-tw//2-pad), max(0, cy- th//2 - pad - 12)
                bx2,by2 = min(w-1, bx1 + tw + 2*pad), min(h-1, by1 + th + 2*pad)
                cv2.rectangle(out,(bx1,by1),(bx2,by2),style["badge"],-1)
                cv2.putText(out, name, (bx1+pad, by2-pad), font, scale, WHITE, 1, cv2.LINE_AA)
    return out

# ---------- Gate & geometry helpers (추가) ----------
def rect_from_ratio(W, H, wr, hr):
    bw, bh = int(W*wr), int(H*hr)
    cx0, cy0 = W//2, H//2
    return (cx0 - bw//2, cy0 - bh//2, bw, bh)

def rect_from_pct(W, H, L, T, R, B):
    x1 = int(np.clip(L, 0, 1) * W)
    y1 = int(np.clip(T, 0, 1) * H)
    x2 = int(np.clip(R, 0, 1) * W)
    y2 = int(np.clip(B, 0, 1) * H)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (x1, y1, x2 - x1, y2 - y1)

def draw_gate_box(frame, rect, color=(0,255,255), thick=2):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thick)

def circle_fully_inside_rect(cx, cy, r, rect):
    x, y, w, h = rect
    x1, y1, x2, y2 = x, y, x+w, y+h
    return (cx - r >= x1) and (cx + r <= x2) and (cy - r >= y1) and (cy + r <= y2)

# ---------- Detection decoders ----------
_GRID_CACHE = {}
def _build_grids(size, strides=(8,16,32)):
    grids, s_all = [], []
    for s in strides:
        ny, nx = size//s, size//s
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        g = np.stack([xv, yv], -1).reshape(-1,2).astype(np.float32)
        grids.append(g); s_all.append(np.full((g.shape[0],1), s, np.float32))
    return np.concatenate(grids,0), np.concatenate(s_all,0)

def _sig(x): return 1/(1+np.exp(-x))

def decode_yolox_raw(pred, size, conf=0.35, num_classes=None):
    C5,N = pred.shape
    if num_classes is None: num_classes = C5-5
    if size not in _GRID_CACHE: _GRID_CACHE[size] = _build_grids(size)
    grid, strides = _GRID_CACHE[size]
    if grid.shape[0] != N:
        M=min(N,grid.shape[0]); pred=pred[:,:M]; N=M
    px,py,pw,ph = pred[0],pred[1],pred[2],pred[3]
    pobj=_sig(pred[4]); pcls=_sig(pred[5:5+num_classes])
    grid=grid[:N]; s=strides[:N,0]
    cx=(_sig(px)+grid[:,0])*s; cy=(_sig(py)+grid[:,1])*s
    w=np.exp(pw)*s; h=np.exp(ph)*s
    x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    cls_idx=np.argmax(pcls,0); scores=pcls[cls_idx,np.arange(N)]*pobj
    keep=scores>=conf
    if not np.any(keep): return np.zeros((0,6),np.float32)
    return np.stack([x1[keep],y1[keep],x2[keep],y2[keep],scores[keep],cls_idx[keep].astype(np.float32)],1)

def decode_detection_output(arr, img_size, conf=0.35):
    dets=[]
    for image_id,label,score,x_min,y_min,x_max,y_max in arr.reshape(-1,7):
        if score>=conf:
            dets.append([x_min*img_size,y_min*img_size,x_max*img_size,y_max*img_size,score,label])
    return np.array(dets,np.float32) if dets else np.zeros((0,6),np.float32)

def decode_box5_plus_cls(out0, out1, size, conf=0.35):
    B,N,C = out0.shape; assert B==1 and C==5
    boxes = out0[0].astype(np.float32)
    clsid = out1[0].astype(np.float32) if (out1.ndim==2 and out1.shape[0]==1) else np.zeros((N,),np.float32)
    if np.all((boxes[:,:4]>=0)&(boxes[:,:4]<=1.5)): boxes[:,:4]*=size
    keep = boxes[:,4] >= conf
    if not np.any(keep): return np.zeros((0,6),np.float32)
    b=boxes[keep]; c=clsid[keep]
    return np.concatenate([b[:,:4], b[:,4:5], c[:,None]],1)

def normalize_dets(arr, size):
    """(N,6) [x1,y1,x2,y2,score,cls] float32로 강제 변환."""
    if arr is None:
        return np.zeros((0,6), np.float32)
    X = np.asarray(arr)
    if X.ndim == 3 and X.shape[0] == 1:
        X = X[0]
    if X.ndim != 2:
        return np.zeros((0,6), np.float32)
    if X.shape[1] >= 7:  # [image_id,label,score,x1,y1,x2,y2]
        coords = np.stack([X[:,3], X[:,4], X[:,5], X[:,6]], 1).astype(np.float32)
        if np.all((coords >= 0.0) & (coords <= 1.5)):
            coords *= float(size)
        score = X[:,2:3].astype(np.float32)
        cls   = X[:,1:2].astype(np.float32)
        Y = np.concatenate([coords, score, cls], 1)
        return Y.astype(np.float32)
    if X.shape[1] == 6:
        Y = X.astype(np.float32)
    elif X.shape[1] == 5:
        score = X[:,4:5].astype(np.float32)
        cls   = np.zeros((X.shape[0],1), np.float32)
        Y = np.concatenate([X[:,:4].astype(np.float32), score, cls], 1)
    elif X.shape[1] == 4:
        score = np.ones((X.shape[0],1), np.float32)
        cls   = np.zeros((X.shape[0],1), np.float32)
        Y = np.concatenate([X[:,:4].astype(np.float32), score, cls], 1)
    else:
        Y = X[:,:6].astype(np.float32)
    if np.all((Y[:,:4] >= 0.0) & (Y[:,:4] <= 1.5)):
        Y[:,:4] *= float(size)
    return Y.astype(np.float32)

# ---------- Seg helpers ----------
def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _prepare_cls_input(bgr, size, scale=255.0):
    # 중앙 정사각형 크롭 → resize → RGB → NCHW float32 [0..1]
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2; x0 = (w - s) // 2
    crop = bgr[y0:y0+s, x0:x0+s]
    img  = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / float(scale or 255.0)
    return rgb.transpose(2,0,1)[None]

def _argmax_logits(arr):
    X = np.asarray(arr)
    X = X.reshape(1, -1) if X.ndim != 2 else X
    X = X - X.max(axis=1, keepdims=True)
    e = np.exp(X); probs = e / e.sum(axis=1, keepdims=True)
    idx = int(np.argmax(probs[0])); conf = float(probs[0, idx])
    return idx, conf

# ---------- ORT helpers ----------
def ort_session(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 0
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=sess_opt, providers=providers)

def ort_run(sess, inputs: dict, out_names=None):
    outs = sess.get_outputs()
    names = [o.name for o in outs] if out_names is None else out_names
    return sess.run(names, inputs)

# ---------- seg-map helpers ----------
def parse_seg_map(s: str, num_classes_hint=3):
    mapping = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        mapping[int(a)] = int(b)
    for i in range(num_classes_hint):
        mapping.setdefault(i, i)
    return mapping

def remap_mask(mask: np.ndarray, m: dict):
    out = np.zeros_like(mask)
    for src, dst in m.items():
        out[mask == src] = dst
    return out

# ---------- Cap circle + stain ratio ----------
def try_hough_circle(bgr_roi):
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                             param1=120, param2=30, minRadius=10, maxRadius=800)
    if circs is None:
        return None
    x, y, r = np.round(circs[0,0]).astype(int)
    return (x, y, r)

def grade_from_ratio(ratio):
    if ratio <= 0.0:
        return (0,255,0)
    elif ratio <= 30.0:
        return (0,255,255)
    else:
        return (0,0,255)

def draw_cap_circle_and_ratio(frame, cx, cy, diameter_px, stain_mask_full, *, stain_id=2):
    """
    기존 함수 이름 유지.
    변경 내용:
      - 삭제/제거 X (연결성 필터, 열림연산 등 전부 제거)
      - '픽셀 수'로만 stain 비율 계산
      - 분모 = 이너서클 픽셀 총계
      - 분자 = 이너서클 내 stain_id 픽셀 수
      - rim(림 제외) 옵션만 유지 (기본 6%)
    """
    import os
    H, W = frame.shape[:2]
    cx = int(np.clip(cx, 0, W - 1))
    cy = int(np.clip(cy, 0, H - 1))

    # 림 제외 비율
    rim_frac = float(os.getenv('STAIN_RIM_FRAC', '0.06'))  # 6%
    r_out = int(max(1, round(diameter_px / 2.0)))
    r_in  = max(1, int(round(r_out * (1.0 - np.clip(rim_frac, 0.0, 0.3)))))

    # ---- 분모(캡 전체) 마스크 생성 ----
    cap_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(cap_mask, (cx, cy), r_in, 255, -1)

    # ---- 분자(stain 픽셀) 계산 (삭제/제거 무시) ----
    stain_bin = (stain_mask_full == stain_id).astype(np.uint8) * 255
    inter = cv2.bitwise_and(stain_bin, cap_mask)

    stain_px = int(cv2.countNonZero(inter))
    cap_px   = int(cv2.countNonZero(cap_mask))
    ratio = float(stain_px) / max(cap_px, 1) * 100.0
    ratio = float(np.clip(ratio, 0.0, 100.0))

    # ---- 오버레이 표시 (GUI용) ----
    color = grade_from_ratio(ratio)
    cv2.circle(frame, (cx, cy), r_out, (200, 120, 255), 1)   # 원 가이드
    cv2.circle(frame, (cx, cy), r_in,  (255,   0, 255), 2)   # 분모 영역
    cv2.line(frame, (max(0, cx - r_in), cy), (min(W - 1, cx + r_in), cy), (255, 0, 255), 2)
    cv2.putText(frame, f"{ratio:.1f}%", (max(0, cx - r_in), min(H - 10, cy + r_in + 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return ratio, stain_px, cap_px

# =========================
# Detection Worker Thread
# =========================
class DetectionWorker(threading.Thread):
    """
    실시간 워커:
      - 프레임에서 '디텍션만' 수행 (세그는 하지 않음)
      - 게이트 박스 안 (full/center) 조건 충족 시 캡처
      - 캡처된 원본 프레임을 snap_q로 전달 (DecideWorker가 Seg/판정)
      - 프리뷰에는 bbox/게이트만 그려줌 (stain% 미표시)
    """
    def __init__(self, get_frame_fn, config, snap_q=None):
        super().__init__(daemon=True)
        self.get_frame = get_frame_fn
        self.cfg = config
        self._stop = threading.Event()
        self.last_annot = None
        self.last_ratio = None  # 실시간 ratio는 제공하지 않음
        self.last_time  = 0.0

        # ONNX Runtime (Det only)
        self.det_sess = ort_session(self.cfg['det_path'])
        self.det_in_name = self.det_sess.get_inputs()[0].name
        self.det_out_names = [o.name for o in self.det_sess.get_outputs()]

        # labels
        self.det_labels = [s.strip() for s in self.cfg['det_labels'].split(",") if s.strip()]

        # gate / capture
        self._gate_rect = None
        self._last_capture_ts = 0.0
        self.snap_q = snap_q if snap_q is not None else queue.Queue(maxsize=16)

        # FPS
        self._fps_t0 = time.time()
        self._fps_n  = 0

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                out = self._process_frame(frame)
                self.last_annot = out
            except Exception:
                self.last_annot = frame
            time.sleep(0.005)

    def _process_frame(self, frame):
        H, W = frame.shape[:2]
        out_frame = frame.copy()

        # ---------- 게이트 직사각형 초기화 ----------
        if self._gate_rect is None:
            roi_pct = self.cfg.get('roi_pct', '')
            box_str = self.cfg.get('box', '')
            cap_box = self.cfg.get('capture_box', '0.3,0.3')
            if roi_pct:
                L,T,R,B = [float(v) for v in roi_pct.split(",")]
                self._gate_rect = rect_from_pct(W, H, L, T, R, B)
            elif box_str:
                bx,by,bw,bh = [int(v) for v in box_str.split(",")]
                self._gate_rect = (bx, by, bw, bh)
            else:
                wr,hr = [float(x) for x in cap_box.split(",")] if "," in cap_box else (0.3,0.3)
                self._gate_rect = rect_from_ratio(W, H, wr, hr)

        # ---------- DET ----------
        d_img, d_r, d_l, d_t = letterbox(frame, self.cfg['det_size'])
        d_rgb = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_ten = d_rgb.transpose(2,0,1)[None].astype(np.float32)
        if self.cfg.get('scale'): d_ten /= self.cfg['scale']

        outs = self.det_sess.run(self.det_out_names, {self.det_in_name: d_ten})
        d_arrs = []
        for a in outs:
            d_arrs.extend(a if isinstance(a, list) else [np.asarray(a)])

        det = None
        if len(d_arrs) >= 2:
            a0, a1 = d_arrs[0], d_arrs[1]
            if a0.ndim==3 and a0.shape[0]==1 and a0.shape[2]==5 and a1.ndim>=2:
                det = decode_box5_plus_cls(a0, a1, self.cfg['det_size'], self.cfg['det_conf'])
            elif a1.ndim==3 and a1.shape[0]==1 and a1.shape[2]==5 and a0.ndim>=2:
                det = decode_box5_plus_cls(a1, a0, self.cfg['det_size'], self.cfg['det_conf'])
        if det is None or (hasattr(det, "size") and det.size==0):
            for a in d_arrs:
                if (a.ndim==2 and a.shape[-1]>=6): det=a; break
                if (a.ndim==3 and a.shape[0]==1 and a.shape[-1]>=6): det=a[0]; break
        if det is None or (hasattr(det, "size") and det.size==0):
            for a in d_arrs:
                if a.ndim==4 and a.shape[-1]==7:
                    det = decode_detection_output(a, self.cfg['det_size'], self.cfg['det_conf'])
                    break
        if det is None or (hasattr(det, "size") and det.size==0):
            for a in d_arrs:
                if a.ndim==3 and a.shape[0]==1 and (a.shape[1]>=6 or a.shape[2]>=6):
                    raw=a[0]
                    pred = raw if (raw.ndim==2 and raw.shape[0]>=6) else (raw.T if (raw.ndim==2 and raw.shape[1]>=6) else None)
                    if pred is None and raw.ndim==3:
                        tmp=raw.reshape(raw.shape[0],-1); pred=tmp if tmp.shape[0]>=6 else tmp.T
                    if pred is not None:
                        det = decode_yolox_raw(pred, self.cfg['det_size'], self.cfg['det_conf'])
                        break
        if det is None:
            det = np.zeros((0,6), np.float32)

        det = normalize_dets(det, self.cfg['det_size'])

        # ---------- 프리뷰: 게이트/박스만 ----------
        draw_gate_box(out_frame, self._gate_rect, (0,255,255), 2)
        if det.shape[0] == 0:
            self._tick_fps(out_frame)
            return out_frame

        # 최고 score 하나 쓰기
        x1,y1,x2,y2,score,cls = map(float, det[np.argmax(det[:,4])][:6])
        gx1=int(np.clip((x1-d_l)/d_r,0,W-1)); gy1=int(np.clip((y1-d_t)/d_r,0,H-1))
        gx2=int(np.clip((x2-d_l)/d_r,0,W-1)); gy2=int(np.clip((y2-d_t)/d_r,0,H-1))
        if gx2<=gx1 or gy2<=gy1:
            self._tick_fps(out_frame)
            return out_frame

        # 박스/라벨만 표시
        draw_box(out_frame, gx1, gy1, gx2, gy2, score, cls, self.det_labels, color=PINK)

        # ---------- 게이트 판정 + 캡처 ----------
        gate_mode = self.cfg.get('gate_mode', 'full')
        dyn_k = float(self.cfg.get('dynamic_diam_k', 0.90))
        dyn_diam = max(10.0, dyn_k * min(gx2-gx1, gy2-gy1))
        r_cap = int(round(dyn_diam / 2.0))

        # 중심점(허프 보정 옵션 없음: Seg가 없으므로 ROI만으로 계산)
        cx, cy = (gx1+gx2)//2, (gy1+gy2)//2
        if gate_mode == "full":
            inside_ok = circle_fully_inside_rect(cx, cy, r_cap, self._gate_rect)
        else:
            gx, gy, gw2, gh2 = self._gate_rect
            inside_ok = (gx <= cx <= gx+gw2 and gy <= cy <= gy+gh2)

        if inside_ok:
            now = time.time()
            cooldown = float(self.cfg.get('capture_cooldown', 0.2))
            if now - self._last_capture_ts >= cooldown:
                ts_ms = int(now * 1000)

                # 옵션: raw 저장
                if self.cfg.get('save_raw_caps', 0):
                    os.makedirs("captures/raw", exist_ok=True)
                    cv2.imwrite(f"captures/raw/{ts_ms}.jpg", frame)

                # 스냅샷 전달 (DecideWorker가 Seg/판정)
                try:
                    self.snap_q.put_nowait((ts_ms, frame.copy()))
                except queue.Full:
                    try: self.snap_q.get_nowait()
                    except: pass
                    self.snap_q.put_nowait((ts_ms, frame.copy()))

                self._last_capture_ts = now

        self._tick_fps(out_frame)
        return out_frame

    def _tick_fps(self, out_frame):
        self._fps_n += 1
        dt = time.time() - self._fps_t0
        if dt >= 0.5:
            self._fps_t0 = time.time()
            self._fps_n = 0

# =========================
# Decide Worker (캡쳐 후 오프라인 판정 + 액츄에이터 콜백)
# =========================
class DecideWorker(threading.Thread):
    """
    오프라인 워커:
      - DetectionWorker가 큐에 넣은 스냅샷을 소비
      - '세그멘테이션 + mask remap + 오염비율 계산 + 등급(OK/NG30/NG80)'
      - 주석이미지 저장 / CSV 로그 / 액츄에이터 콜백
    """
    def __init__(self, snap_q: "queue.Queue", cfg: dict, on_decision_cb):
        super().__init__(daemon=True)
        self.snap_q = snap_q
        self.cfg = cfg
        self.on_decision = on_decision_cb
        self._stop = threading.Event()

        # ONNX Runtime (Seg + Det 재사용: 스냅샷에서도 1회 det 수행 → ROI 잡아 Seg)
        self.det_sess = ort_session(self.cfg['det_path'])
        self.seg_sess = ort_session(self.cfg['seg_path'])
        self.det_in = self.det_sess.get_inputs()[0].name
        self.seg_in = self.seg_sess.get_inputs()[0].name
        self.det_outs = [o.name for o in self.det_sess.get_outputs()]
        self.seg_out  = self.seg_sess.get_outputs()[0].name

        # label & seg map
        self.det_labels = [s.strip() for s in self.cfg['det_labels'].split(",") if s.strip()]
        self.seg_labels = [s.strip() for s in self.cfg['seg_labels'].split(",") if s.strip()]
        self.seg_map    = parse_seg_map(self.cfg['seg_map'], num_classes_hint=max(3, len(self.seg_labels) or 3))  # snap 스타일:contentReference[oaicite:12]{index=12}
        self.stain_disp_id = self.seg_map.get(self.cfg['stain_id'], self.cfg['stain_id'])
        self.sticker_disp_id = self.seg_map.get(self.cfg['sticker_id'], self.cfg['sticker_id'])

        # --- [추가] 색상 분류 세션(옵션) ---
        self.cls_sess = None
        self.cls_in = None
        self.cls_out = None
        self.cls_labels = [s.strip() for s in (self.cfg.get('cls_labels') or '').split(',') if s.strip()]
        self.cls_size = int(self.cfg.get('cls_size', 224))
        self.expected_color = (self.cfg.get('expected_color') or '').strip().lower() or None
        self.cls_map = parse_seg_map(self.cfg.get('cls_map',''), num_classes_hint=max(3, len(self.cls_labels) or 3))
        try:
            cls_path = self.cfg.get('cls_path')
            if cls_path and os.path.isfile(cls_path):
                self.cls_sess = ort_session(cls_path)
                self.cls_in   = self.cls_sess.get_inputs()[0].name
                self.cls_out  = self.cls_sess.get_outputs()[0].name
        except Exception:
            self.cls_sess = None
    
    def stop(self):
        self._stop.set()

    def run(self):
        # 로그/이미지 디렉토리
        log_dir = "captures/log"
        os.makedirs(log_dir, exist_ok=True)
        log_path_v2 = os.path.join(log_dir, "decisions_v2.csv")
        if self.cfg.get('log_csv', 1):
            if not os.path.isfile(log_path_v2):
                with open(log_path_v2, "w") as f:
                    # v2: ts,grade,color,color_conf,sticker,stain_pct
                    f.write("ts,grade,color,color_conf,sticker,stain_pct\n")
        if self.cfg.get('save_annotated', 1):
            os.makedirs("captures/annotated", exist_ok=True)

        print("[DecideWorker] Warming up ONNX sessions...")
        try:
            # 1. Segmentation Model Warm-up (가장 무거울 가능성 높음)
            # seg_size에 맞는 더미 입력 생성
            dummy_seg_input = np.zeros(
                (1, 3, self.cfg['seg_size'], self.cfg['seg_size']), 
                dtype=np.float32
            )
            self.seg_sess.run([self.seg_out], {self.seg_in: dummy_seg_input})
            print("[DecideWorker] Segmentation model warmed up.")
            
            # 2. Detection Model Warm-up (DecideWorker에서도 사용)
            dummy_det_input = np.zeros(
                (1, 3, self.cfg['det_size'], self.cfg['det_size']), 
                dtype=np.float32
            )
            self.det_sess.run(self.det_outs, {self.det_in: dummy_det_input})
            print("[DecideWorker] Detection model warmed up.")

            # 3. Classification Model Warm-up (옵션, 있다면)
            if self.cls_sess is not None:
                dummy_cls_input = np.zeros(
                    (1, 3, self.cls_size, self.cls_size), 
                    dtype=np.float32
                )
                self.cls_sess.run([self.cls_out], {self.cls_in: dummy_cls_input})
                print("[DecideWorker] Classification model warmed up.")
                
        except Exception as e:
            print(f"[DecideWorker] WARNING: Failed to warm up session: {e}")
        # ---------------------------------------------------

        while not self._stop.is_set():
            try:
                ts_ms, snap_bgr = self.snap_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                grade, ratio, annotated, has_sticker, pred_color, pred_conf = self._analyze_one(snap_bgr)

                # 저장 파일명에 color 태그 포함
                if self.cfg.get('save_annotated', 1):
                    color_tag   = f"color-{pred_color}" if pred_color else "color-NA"
                    sticker_tag = "yes" if has_sticker else "no"
                    outp = f"captures/annotated/{grade}_{color_tag}_sticker-{sticker_tag}_{ratio:.1f}pct_{ts_ms}.jpg"
                    cv2.imwrite(outp, annotated)

                # CSV(v2): ts,grade,color,color_conf,sticker,stain_pct
                if self.cfg.get('log_csv', 1):
                    cc = f"{pred_conf:.3f}" if (pred_conf is not None) else ""
                    with open(log_path_v2, "a") as f:
                        f.write(f"{ts_ms},{grade},{pred_color or ''},{cc},{'1' if has_sticker else '0'},{ratio:.3f}\n")

                # 기존 콜백(시그니처 그대로)
                if callable(self.on_decision):
                    self.on_decision(grade, ratio, has_sticker)
            except Exception as e:
                pass

    def _analyze_one(self, img_bgr):
        """
        snap_after_seg3.py의 analyze_one_frame 로직을 run.py 스타일로 이식
        (det → ROI → seg → mask remap → 비율 계산 → 등급)
        """
        H, W = img_bgr.shape[:2]

        # ---------- gate (표시용) ----------
        if self.cfg.get('roi_pct'):
            L,T,R,B = [float(v) for v in self.cfg['roi_pct'].split(",")]
            gate = rect_from_pct(W, H, L, T, R, B)
        elif self.cfg.get('box'):
            bx,by,bw,bh = [int(v) for v in self.cfg['box'].split(",")]
            gate = (bx,by,bw,bh)
        else:
            wr,hr = [float(x) for x in self.cfg.get('capture_box','0.3,0.3').split(",")]
            gate = rect_from_ratio(W, H, wr, hr)

        annotated = img_bgr.copy()
        draw_gate_box(annotated, gate, (0,255,255), 2)

        # ---------- det (스냅샷에서 다시 한 번) ----------
        d_img, d_r, d_l, d_t = letterbox(img_bgr, self.cfg['det_size'])
        d_rgb = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d = d_rgb.transpose(2,0,1)[None].astype(np.float32)
        if self.cfg.get('scale'): d /= self.cfg['scale']
        outs = self.det_sess.run(self.det_outs, {self.det_in:d})
        arrs=[]; [arrs.extend(a if isinstance(a,list) else [np.asarray(a)]) for a in outs]

        det=None
        if len(arrs)>=2:
            a0,a1=arrs[0],arrs[1]
            if a0.ndim==3 and a0.shape[0]==1 and a0.shape[2]==5 and a1.ndim>=2:
                det=decode_box5_plus_cls(a0,a1,self.cfg['det_size'],self.cfg['det_conf'])
            elif a1.ndim==3 and a1.shape[0]==1 and a1.shape[2]==5 and a0.ndim>=2:
                det=decode_box5_plus_cls(a1,a0,self.cfg['det_size'],self.cfg['det_conf'])
        if det is None or (hasattr(det, "size") and det.size==0):
            for a in arrs:
                if (a.ndim==2 and a.shape[-1]>=6): det=a; break
                if (a.ndim==3 and a.shape[0]==1 and a.shape[-1]>=6): det=a[0]; break
        if det is None or (hasattr(det, "size") and det.size==0):
            for a in arrs:
                if a.ndim==4 and a.shape[-1]==7:
                    det=decode_detection_output(a,self.cfg['det_size'],self.cfg['det_conf']); break
        if det is None or (hasattr(det, "size") and det.size==0):
            for a in arrs:
                if a.ndim==3 and a.shape[0]==1 and (a.shape[1]>=6 or a.shape[2]>=6):
                    raw=a[0]; pred=raw if (raw.ndim==2 and raw.shape[0]>=6) else (raw.T if (raw.ndim==2 and raw.shape[1]>=6) else None)
                    if pred is None and raw.ndim==3:
                        tmp=raw.reshape(raw.shape[0],-1); pred=tmp if tmp.shape[0]>=6 else tmp.T
                    if pred is not None: det=decode_yolox_raw(pred,self.cfg['det_size'],self.cfg['det_conf']); break
        if det is None:
            det=np.zeros((0,6),np.float32)

        if det.shape[0]==0:
            _label_badge(annotated, 12, 40, "NO_CAP", (40,40,40), WHITE)
            return "NO_CAP", 0.0, annotated, False,None,None  # [PATCH] has_sticker=False

        # 최고 score 1개 ROI
        x1,y1,x2,y2,score,cls = map(float, det[np.argmax(det[:,4])][:6])
        gx1=int(np.clip((x1-d_l)/d_r,0,W-1)); gy1=int(np.clip((y1-d_t)/d_r,0,H-1))
        gx2=int(np.clip((x2-d_l)/d_r,0,W-1)); gy2=int(np.clip((y2-d_t)/d_r,0,H-1))
        if gx2<=gx1 or gy2<=gy1:
            return "NO_CAP", 0.0, annotated, False, None, None
        roi = annotated[gy1:gy2, gx1:gx2]
        if roi.size==0:
            return "NO_CAP", 0.0, annotated, False, None, None
        
        # ---------- [추가] 색상 분류 ----------
        pred_color = None; pred_conf = None
        if self.cls_sess is not None:
            try:
                cls_in = _prepare_cls_input(roi, self.cls_size, scale=(self.cfg.get('scale') or 255.0))
                logits = self.cls_sess.run([self.cls_out], {self.cls_in: cls_in})[0]
                raw_idx, conf = _argmax_logits(logits)
                idx = self.cls_map.get(int(raw_idx), int(raw_idx)) if hasattr(self, 'cls_map') and isinstance(self.cls_map, dict) else int(raw_idx)
                pred_color = (self.cls_labels[idx] if 0 <= idx < len(self.cls_labels) else str(idx)).lower()
                pred_color = (pred_color or 'unknown').lower()
                pred_conf  = conf
            except Exception:
                pass
        # ---------- seg ----------
        s_img, s_r, s_l, s_t = letterbox(roi, self.cfg['seg_size'])
        s_rgb = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
        s = s_rgb.transpose(2,0,1)[None].astype(np.float32)
        if self.cfg.get('scale'): s /= self.cfg['scale']
        s_res = self.seg_sess.run([self.seg_out], {self.seg_in:s})[0]

        # multi-class / binary 대응
        if s_res.ndim==4 and s_res.shape[1]>1:
            prob = softmax(s_res, axis=1)[0]
            mask = np.argmax(prob, axis=0).astype(np.uint8)
        else:
            mask = (s_res[0,0] > 0.5).astype(np.uint8)

        # 표기가 원하는 class id에 맞게 remap
        mask = remap_mask(mask, self.seg_map)

        # letterbox 되돌리기 (ROI 크기와 정합)
        canvas = np.zeros((self.cfg['seg_size'], self.cfg['seg_size']), dtype=np.uint8)
        nh = int(round(roi.shape[0]*s_r)); nw = int(round(roi.shape[1]*s_r))
        top = (self.cfg['seg_size']-nh)//2; left=(self.cfg['seg_size']-nw)//2
        mask_rs = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
        canvas[top:top+nh, left:left+nw] = mask_rs
        mask_roi = cv2.resize(canvas,(roi.shape[1],roi.shape[0]),interpolation=cv2.INTER_NEAREST)

        # ROI 오버레이
        annotated[gy1:gy2, gx1:gx2] = draw_seg_on_roi(
            roi, mask_roi, self.seg_labels,
            sticker_cid=self.sticker_disp_id,
            stain_cid=self.stain_disp_id
        )

        # ---------- [추가] COLOR 배지(프리뷰/저장 공통) ----------
        color_mismatch = False
        if self.expected_color and pred_color:
            color_mismatch = (pred_color != self.expected_color)

        if pred_color:
            badge_txt = f"COLOR: {pred_color.upper()}"
            if pred_conf is not None:
                badge_txt += f" ({pred_conf*100:.0f}%)"
            _label_badge(
                annotated,
                max(0, gx1), max(0, gy1-36),
                badge_txt,
                (0,0,255) if color_mismatch else (80,80,80),
                WHITE
            )
        # ---------- STICKER 여부 판단 + 상단 배지 출력 [PATCH] ----------
        has_sticker = bool(np.any(mask_roi == self.sticker_disp_id))
        badge_text = f"STICKER: {'YES' if has_sticker else 'NO'}"
        badge_bg   = (40,180,40) if has_sticker else (50,50,200)
        # 박스 상단 중앙 쯤에 뱃지
        bx = max(12, gx1 + (gx2 - gx1)//2 - 80)
        by = max(24, gy1 - 10)
        _label_badge(annotated, bx, by, badge_text, badge_bg, WHITE)

        # ---------- 오염 비율 (원형 캡 내 비율) ----------
        # 동적 지름: 디텍션 박스 기반 or 허프 보정
        dyn_k = float(self.cfg.get('dynamic_diam_k', 0.90))
        dyn_diam = max(10.0, dyn_k * min(gx2-gx1, gy2-gy1))
        cx, cy = (gx1+gx2)//2, (gy1+gy2)//2
        if self.cfg.get('use_hough'):
            hc = try_hough_circle(annotated[gy1:gy2, gx1:gx2])
            if hc is not None:
                rx, ry, rr = hc
                cx, cy = gx1 + rx, gy1 + ry
                dyn_diam = max(dyn_diam, rr * 2.0)

        stain_full = np.zeros((H,W), np.uint8)
        stain_full[gy1:gy2, gx1:gx2] = np.where(mask_roi==self.stain_disp_id, self.stain_disp_id, 0)

        # [PATCH] 픽셀 카운트도 받는다
        ratio, stain_px, cap_px = draw_cap_circle_and_ratio(
            annotated, cx, cy, dyn_diam, stain_full, stain_id=self.stain_disp_id
        )
        ratio = float(np.clip(ratio, 0.0, 100.0))
        
        # [PATCH] 좌측 상단에 메트릭 표기
        sticker_line = f"sticker: {'YES' if has_sticker else 'NO'}"
        lines  = [
            sticker_line,
            f"cap_px:   {cap_px}",
            f"stain_px: {stain_px}",
            f"stain:    {ratio:.1f}%",
        ]
        colors = [
            (40,180,40) if has_sticker else (50,50,200),
            (255,255,255),
            (255,255,255),
            (255,255,255),
        ]
        # ---------- [추가] 메트릭 첫 줄에 color ----------
        if pred_color:
            # 숫자일 경우 라벨 이름으로 변환
            if isinstance(pred_color, (int, np.integer)) and hasattr(self, "cls_labels"):
                if 0 <= pred_color < len(self.cls_labels):
                    pred_color = self.cls_labels[pred_color]
                else:
                    pred_color = str(pred_color)
            lines.insert(0, f"color: {pred_color} ({pred_conf*100:.0f}% )" if pred_conf is not None else f"color: {pred_color}")
            colors.insert(0, (0,0,255) if color_mismatch else (255,255,255))

        put_lines_top_left(annotated, lines, colors)

        # ---------- 등급 판정 ----------
        if color_mismatch:
            grade = "NG80"            # 색상 불일치 = 완전불량(강제)
        elif not has_sticker:
            grade = "NG_STICKER"      # 스티커 없음 = 불량
        elif ratio <= 0.0:
            grade = "OK"
        elif ratio <= 30.0:
            grade = "NG30"
        else:
            grade = "NG80"

        # 중앙 배지: 등급 | STAIN xx.x%
        _label_badge(annotated, max(0,cx-60), max(0,cy-80), f"{grade} | STAIN {ratio:.1f}%", (40,40,40), WHITE)
        draw_box(annotated, gx1, gy1, gx2, gy2, score, cls, self.det_labels, color=PINK)

        return grade, ratio, annotated, has_sticker, pred_color, pred_conf   # [PATCH] sticker flag 반환


# =========================
# Hardware / DB / Serial
# =========================
class HW:
    """FactoryController 어댑터: 릴레이 8개 + 센서 4개 + 시스템 시작/정지"""
    def __init__(self, port=None, debug=True):
        self.ctrl = None
        self.is_dummy = True
        self.dev_name = "dummy"
        self.state = {i: False for i in range(1,9)}
        if FactoryController:
            try:
                self.ctrl = FactoryController(port=port, debug=debug)
                self.is_dummy = self.ctrl.is_dummy
                self.dev_name = getattr(self.ctrl, "_FactoryController__device_name", "unknown")
            except Exception:
                self.ctrl = None
        self.DEV_ON  = getattr(FactoryController, "DEV_ON", False)
        self.DEV_OFF = getattr(FactoryController, "DEV_OFF", True)

        self.relay_map = {
            1: getattr(Outputs, "BEACON_RED",     None),
            2: getattr(Outputs, "BEACON_ORANGE",  None),
            3: getattr(Outputs, "BEACON_GREEN",   None),
            4: getattr(Outputs, "BEACON_BUZZER",  None),
            5: getattr(Outputs, "LED",            None),
            6: getattr(Outputs, "CONVEYOR_EN",    None),
            7: getattr(Outputs, "ACTUATOR_1",     None),
            8: getattr(Outputs, "ACTUATOR_2",     None),
        }
        self.pwm_pin = getattr(Outputs, "CONVEYOR_PWM", None)

    def set_relay(self, idx:int, on:bool):
        self.state[idx] = on
        if not self.ctrl: return
        pin = self.relay_map.get(idx)
        if pin is None: return
        dev = getattr(self.ctrl, "_FactoryController__device", None)
        if dev is None: return

        # Conveyor EN(active-low), PWM 연동
        if pin == getattr(Outputs, "CONVEYOR_EN", -1):
            level = self.DEV_OFF if on else self.DEV_ON
            try:
                dev.set(pin, level)
                if self.pwm_pin is not None:
                    dev.set(self.pwm_pin, 0 if on else 255)
            except Exception:
                pass
            return

        try:
            dev.set(pin, self.DEV_ON if on else self.DEV_OFF)
        except Exception:
            pass

    def system_start(self):
        if hasattr(self.ctrl, "system_start"):
            try: self.ctrl.system_start()
            except Exception: pass

    def system_stop(self):
        if hasattr(self.ctrl, "system_stop"):
            try: self.ctrl.system_stop()
            except Exception: pass

    def get_sensors(self):
        res = {"start_button": False, "stop_button": False, "sensor_1": False, "sensor_2": False}
        if not self.ctrl: return res
        dev = getattr(self.ctrl, "_FactoryController__device", None)
        name = getattr(self.ctrl, "_FactoryController__device_name", "")
        if name == "arduino" and dev is not None and Inputs:
            try:
                res["start_button"] = (dev.get(Inputs.START_BUTTON) == 0)
                res["stop_button"]  = (dev.get(Inputs.STOP_BUTTON)  == 0)
                res["sensor_1"]     = (dev.get(Inputs.PHOTOELECTRIC_SENSOR_1) == 0)
                res["sensor_2"]     = (dev.get(Inputs.PHOTOELECTRIC_SENSOR_2) == 0)
            except Exception:
                pass
        return res

    def get_outputs(self):
        out = {}
        if not self.ctrl:
            return dict(self.state)
        dev = getattr(self.ctrl, "_FactoryController__device", None)
        name = getattr(self.ctrl, "_FactoryController__device_name", "")
        for idx, pin in self.relay_map.items():
            val = None
            if name == "arduino" and dev is not None and pin is not None:
                try:
                    raw = dev.get(pin)
                    if raw in (0,1,True,False):
                        # active-low: 0이면 ON (단, EN은 반대)
                        val = (raw == 0) if pin != getattr(Outputs,"CONVEYOR_EN",-1) else (raw == 1)
                except Exception:
                    val = None
            out[idx] = self.state[idx] if val is None else bool(val)
        return out

    def close(self):
        try:
            if self.ctrl: self.ctrl.close()
        except Exception:
            pass

# ==== Env / Config ====
DB_HOST = os.getenv('DB_HOST', '10.10.14.17')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '2453.')
DB_NAME = os.getenv('DB_NAME', 'iotdb')
DB_CHARSET = 'utf8mb4'

DB_TABLE_6 = os.getenv('DB_TABLE_6', 'qc_agg6')   # ts, id, m1_good,m1_sev,m1_par,m2_good,m2_sev,m2_par
PARTIAL_WEIGHT = float(os.getenv('PARTIAL_WEIGHT', '0.3'))

# Camera1
CAM_INDEX   = int(os.getenv('CAM_INDEX', '0'))
CAM_REQ_W   = int(os.getenv('CAM_REQ_W', '640'))
CAM_REQ_H   = int(os.getenv('CAM_REQ_H', '480'))
CAM_VIEW_W  = int(os.getenv('CAM_VIEW_W', '560'))
CAM_VIEW_H  = int(os.getenv('CAM_VIEW_H', '315'))
CAM_FPS     = int(os.getenv('CAM_FPS', '30'))
CAM_BUFFERS = int(os.getenv('CAM_BUFFERS', '1'))
CAM_FOURCC  = os.getenv('CAM_FOURCC', 'MJPG')
CAM_BACKEND = os.getenv('CAM_BACKEND', '')
RESAMPLE    = os.getenv('RESAMPLE', 'LANCZOS')

# Camera2
CAM2_INDEX   = int(os.getenv('CAM2_INDEX', '2'))
CAM2_REQ_W   = int(os.getenv('CAM2_REQ_W', '640'))
CAM2_REQ_H   = int(os.getenv('CAM2_REQ_H', '480'))
CAM2_VIEW_W  = int(os.getenv('CAM2_VIEW_W', str(CAM_VIEW_W)))
CAM2_VIEW_H  = int(os.getenv('CAM2_VIEW_H', str(CAM_VIEW_H)))
CAM2_FPS     = int(os.getenv('CAM2_FPS', str(CAM_FPS)))
CAM2_FOURCC  = os.getenv('CAM2_FOURCC', 'MJPG')
CAM2_BACKEND = os.getenv('CAM2_BACKEND', '')

def parse_window(win_str):
    try:
        if win_str.endswith('m'): return timedelta(minutes=int(win_str[:-1]))
        if win_str.endswith('h'): return timedelta(hours=int(win_str[:-1]))
        if win_str.endswith('d'): return timedelta(days=int(win_str[:-1]))
    except Exception:
        pass
    return timedelta(minutes=5)

# ==== DB helper ====
def db_query(sql, args=None):
    if pymysql is None:
        raise RuntimeError("pymysql not installed. pip install pymysql")
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS,
                           db=DB_NAME, charset=DB_CHARSET, autocommit=True)
    try:
        with conn.cursor() as c:
            c.execute(sql, args or ())
            if sql.strip().lower().startswith("select"):
                rows = c.fetchall()
                return rows, [d[0] for d in c.description]
            else:
                return (), []
    finally:
        conn.close()

# ==== Camera Thread (dual) ====
class CameraThread(threading.Thread):
    def __init__(self, index=0, req_w=None, req_h=None, fps=None, fourcc=None, buffers=None, backend=None):
        super().__init__(daemon=True)
        self.index=index; self.frame=None; self._stop=threading.Event(); self.cap=None
        self._lock = threading.Lock()
        self._req_w = req_w or CAM_REQ_W
        self._req_h = req_h or CAM_REQ_H
        self._fps   = fps   or CAM_FPS
        self._fourcc = (fourcc or CAM_FOURCC)[:4]
        self._buffers = buffers if buffers is not None else CAM_BUFFERS
        self._backend = (backend or CAM_BACKEND).upper()

    def _open(self):
        if cv2 is None: return None
        backend_map={'V4L2': cv2.CAP_V4L2, 'DSHOW': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF}
        backend_flag = backend_map.get(self._backend, 0)

        cap = cv2.VideoCapture(self.index, backend_flag) if backend_flag else cv2.VideoCapture(self.index)
        if not (cap and cap.isOpened()):
            for flag in [cv2.CAP_V4L2, cv2.CAP_MSMF, cv2.CAP_DSHOW]:
                cap = cv2.VideoCapture(self.index, flag)
                if cap and cap.isOpened():
                    break
        if not (cap and cap.isOpened()): return None

        try: cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffers)
        except Exception: pass
        try:
            fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._req_w)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_h)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FPS, self._fps)
        except Exception: pass

        time.sleep(0.1)
        ok, f = cap.read()
        if not ok or f is None:
            try: cap.release()
            except Exception: pass
            return None
        return cap

    def run(self):
        while not self._stop.is_set():
            cap = self._open()
            self.cap = cap
            if not cap:
                time.sleep(1.5); continue
            interval = 1.0 / max(5, self._fps)
            while not self._stop.is_set():
                if not cap.grab():
                    time.sleep(0.05)
                    ok, _ = cap.read()
                    if not ok:
                        try: cap.release()
                        except Exception: pass
                        break
                ok, frame = cap.retrieve()
                if ok:
                    with self._lock:
                        self.frame = frame
                else:
                    try: cap.release()
                    except Exception: pass
                    break
                time.sleep(interval * 0.5)

    def get_frame(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._stop.set()
        try:
            if self.cap: self.cap.release()
        except Exception: pass

# ====== 작은 유틸 ======
def _has_model(cfg: dict) -> bool:
    return os.path.isfile(cfg.get('det_path','')) and os.path.isfile(cfg.get('seg_path',''))

# =========================
# App (Tk GUI)
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Factory QC — Dual Camera + DB(6-cols) + Conveyor + Device Panel (overlay + delayed pulse)")
        self.geometry("1360x980")

        # 신호등(비콘) 상태 추적
        self._last_beacon = 'none'
        self._beacon_after = None
        self._beacon_busy = False  # 부팅 시퀀스 중에는 비콘 제어 잠금

        # 하드웨어 컨트롤러
        self.hw = HW(port=None, debug=True)

        # --- 시스템 실행 상태 / arming ---
        self._is_running = False           # RUN/STOP 램프용
        self.active_model = 0              # 0=none, 1=M1, 2=M2

        # 컨트롤러 초기화 직후 강제 STOP (시작 깜빡임 방지)
        try:
            if self.hw:
                self.hw.system_stop()
                self.hw.set_relay(6, False)  # Conveyor OFF
        except Exception:
            pass
        
        # 부팅 인디케이터: RED → ORANGE → GREEN → ALL OFF
        try:
            self._beacon_boot_chase(step_ms=250, hold_ms=1000)
        except Exception:
            pass
        try:
            if self.hw:
                for idx in (1, 2, 3, 4, 5, 7, 8):  # 1=R,2=O,3=G,4=Buzzer,5=LED,7/8=Actuators
                    self.hw.set_relay(idx, False)
        except Exception:
            pass

        # 시작 후 1초 지연 기간(arming): 자동동작/버튼 RUN 차단
        self._armed = False
        self.after(1000, lambda: setattr(self, "_armed", True))

        # Top bar
        top = ttk.Frame(self); top.pack(fill='x', padx=8, pady=6)
        try:
            from PIL import Image, ImageTk
            logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path).resize((42, 30))
                self._logo_tk = ImageTk.PhotoImage(logo_img)
                ttk.Label(top, image=self._logo_tk).pack(side="left", padx=(2, 10))
            else:
                ttk.Label(top, text="LOGO", font=("Arial", 12, "bold")).pack(side="left", padx=(2, 10))
        except Exception as e:
            pass

        self.btn_m1 = ttk.Button(top, text="Model 1 · RUN", command=lambda:self.on_run(1))
        self.btn_m2 = ttk.Button(top, text="Model 2 · RUN", command=lambda:self.on_run(2))
        self.btn_stop = ttk.Button(top, text="STOP", command=self.on_stop)
        self.status_lbl = ttk.Label(top, text="대기")
        ttk.Label(top, text="RUN:").pack(side='left', padx=(0,6))
        self.win_var = tk.StringVar(value='5m')
        self.win_box = ttk.Combobox(top, textvariable=self.win_var, values=['5m','30m','1h','1d'], width=6, state='readonly')
        self.btn_m1.pack(side='left', padx=4); self.btn_m2.pack(side='left', padx=4); self.btn_stop.pack(side='left', padx=4)
        self.win_box.pack(side='left'); ttk.Label(top, text="  상태:").pack(side='left', padx=(16,4)); self.status_lbl.pack(side='left')

        # --- [ADMIN] Hidden toggle hotspot & keybinding ---
        self._admin_visible = False
        self._admin_win = None  # Toplevel 창 캐시
        self._hidden_hotspot = tk.Label(top, text=" ", background=self.cget("bg"))
        self._hidden_hotspot.place(relx=1.0, x=-2, y=2, anchor="ne")
        self._hidden_hotspot.config(width=1, height=1)
        self._hidden_hotspot.bind("<Button-1>", lambda e: self._toggle_admin_panel())
        self.bind_all("<Control-Alt-a>", lambda e: self._toggle_admin_panel())

        # 모델별 상태 라벨
        self.m1_state = tk.StringVar(value='대기')
        self.m2_state = tk.StringVar(value='대기')
        ttk.Label(top, text=" |    M1:").pack(side='left', padx=(14,2))
        ttk.Label(top, textvariable=self.m1_state).pack(side='left')
        ttk.Label(top, text="  M2:").pack(side='left', padx=(8,2))
        ttk.Label(top, textvariable=self.m2_state).pack(side='left')

        # Grid layout
        root = ttk.Frame(self); root.pack(fill='both', expand=True, padx=8, pady=6)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)   # 카메라 영역
        root.grid_rowconfigure(1, weight=3)   # 모델/최근(오른쪽) 크게
        root.grid_rowconfigure(2, weight=1)   # 장치 패널

        # KPI 확장용 변수
        self.k_defw_abs = tk.StringVar(value='-')
        self.k_tput     = tk.StringVar(value='-')
        self.k_updated  = tk.StringVar(value='-')

        # 오버레이에 같이 표기할 실시간 stain% 라벨
        self.r1_var = tk.StringVar(value='CAM1: - %')
        self.r2_var = tk.StringVar(value='CAM2: - %')

        # KPI
        kpi = ttk.LabelFrame(root, text="통합 KPI")
        kpi.grid(row=0, column=0, sticky='nsew', padx=(0,6), pady=(0,6))
        self.var_total = tk.StringVar(value='-'); self.var_good = tk.StringVar(value='-')
        self.var_defw  = tk.StringVar(value='-'); self.var_rate = tk.StringVar(value='-')
        g = ttk.Frame(kpi); g.pack(fill='x', padx=8, pady=8)
        self._row(g,0,"총 처리", self.var_total); self._row(g,1,"정상품", self.var_good)
        self._row(g,2,"불량", self.var_defw); self._row(g,3,"불량률", self.var_rate)
        self._row(g,4,"불량(가중)", self.k_defw_abs)
        self._row(g,5,"처리속도",   self.k_tput)
        self._row(g,6,"업데이트",   self.k_updated)
        ttk.Label(kpi, textvariable=self.r1_var).pack(anchor="w", padx=10)
        ttk.Label(kpi, textvariable=self.r2_var).pack(anchor="w", padx=10)
        self.pb = ttk.Progressbar(kpi, orient='horizontal', length=400, mode='determinate', maximum=100)
        self.pb.pack(padx=8, pady=(4,10))

        # Cameras (overlay 표시)
        camf = ttk.LabelFrame(root, text="USB 카메라")
        camf.grid(row=0, column=1, sticky='nsew', padx=(6,0), pady=(0,6))
        camgrid = ttk.Frame(camf); camgrid.pack(fill='both', expand=True, padx=6, pady=6)
        camgrid.grid_columnconfigure(0, weight=1); camgrid.grid_columnconfigure(1, weight=1)
        self.cam1_lbl = ttk.Label(camgrid, text="CAM1 초기화 중...")
        self.cam2_lbl = ttk.Label(camgrid, text="CAM2 초기화 중...")
        self.cam1_lbl.grid(row=0, column=0, padx=4, pady=4, sticky='nsew')
        self.cam2_lbl.grid(row=0, column=1, padx=4, pady=4, sticky='nsew')

        self.cam1 = CameraThread(index=CAM_INDEX,
                                 req_w=CAM_REQ_W, req_h=CAM_REQ_H, fps=CAM_FPS,
                                 fourcc=CAM_FOURCC, backend=CAM_BACKEND)
        self.cam2 = CameraThread(index=CAM2_INDEX,
                                 req_w=CAM2_REQ_W, req_h=CAM2_REQ_H, fps=CAM2_FPS,
                                 fourcc=CAM2_FOURCC, backend=CAM2_BACKEND)
        self.cam1.start(); self.cam2.start()

        # Detection workers: CAM1(모델1), CAM2(모델2)
        self.det_cfg_base = {
            # 모델 2 입니다.
            # --- [추가] 색상 분류(classification) 관련 ---
            'cls_path':   os.path.expanduser(os.getenv('CLASS_MODEL',  os.path.join(os.path.dirname(os.path.abspath(__file__)), "class.onnx"))),
            'cls_labels': os.getenv('CLASS_LABELS', 'pink,purple'),
            'cls_size':   int(os.getenv('CLASS_SIZE', '224')),
            'cls_map': os.getenv('CLASS_MAP', '1:0,2:1'),
            'expected_color': None,   # 개별 모델 cfg에서 채움 (model1=pink, model2=purple)
            'det_path':  os.path.expanduser(os.getenv('DET_MODEL',  'det.onnx')),
            'seg_path':  os.path.expanduser(os.getenv('SEG_MODEL',  'seg.onnx')),
            'det_size':  int(os.getenv('DET_SIZE', '640')),
            'seg_size':  int(os.getenv('SEG_SIZE', '512')),
            'det_conf':  float(os.getenv('DET_CONF', '0.35')),
            'det_labels': os.getenv('DET_LABELS', 'cap'),
            'seg_labels': os.getenv('SEG_LABELS', 'background,sticker,stain'),
            'seg_map':    os.getenv('SEG_MAP', '0:0,1:2,2:1'),
            'scale':      float(os.getenv('SCALE', '255.0')),
            'cap_diam':   float(os.getenv('CAP_DIAM', '170')),
            'stain_id':   int(os.getenv('STAIN_ID', '1')),
            'sticker_id': int(os.getenv('STICKER_ID', '2')),
            'use_hough':  bool(int(os.getenv('USE_HOUGH', '1'))),
            
            # --- [추가] 게이트/캡쳐 관련 ---
            'roi_pct':   os.getenv('ROI_PCT', '0.3, 0.3, 0.7, 0.7'),                  # "L,T,R,B" (0~1)
            'box':       os.getenv('BOX', ''),                      # "x,y,w,h" (px)
            'capture_box': os.getenv('CAPTURE_BOX', '0.3,0.3'),     # 중앙 비율 (백업)
            'gate_mode': os.getenv('GATE_MODE', 'full'),            # full|center
            'capture_cooldown': float(os.getenv('CAPTURE_COOLDOWN', '0.2')),
            'dynamic_diam_k': float(os.getenv('DYN_DIAM_K', '0.90')),  # 검출박스 기반 지름 스케일
            
            'save_raw_caps': int(os.getenv('SAVE_RAW_CAPS', '0')),   # 게이트 통과시 원본컷 저장
            'save_annotated': int(os.getenv('SAVE_ANNOTATED', '1')), # 판정 결과 이미지 저장
            'log_csv': int(os.getenv('LOG_CSV', '1')),               # 판정 CSV 로깅
        }

        model_dir = os.path.dirname(os.path.abspath(__file__))
        DEFAULT_DET_LOCAL = os.path.join(model_dir, "det.onnx")
        DEFAULT_SEG_LOCAL = os.path.join(model_dir, "seg.onnx")

      # Model 1 (핑크 전용)
        self.det_cfg_m1 = dict(self.det_cfg_base)
        self.det_cfg_m1['det_path'] = os.path.expanduser(os.getenv('DET_MODEL_M1', os.getenv('DET_MODEL', DEFAULT_DET_LOCAL)))
        self.det_cfg_m1['seg_path'] = os.path.expanduser(os.getenv('SEG_MODEL_M1', os.getenv('SEG_MODEL', DEFAULT_SEG_LOCAL)))
        self.det_cfg_m1['cls_path'] = os.path.expanduser(os.getenv('CLASS_MODEL_M1', self.det_cfg_base['cls_path']))
        self.det_cfg_m1['expected_color'] = os.getenv('EXPECTED_COLOR_M1', 'pink')

        # Model 2 (퍼플 전용)
        self.det_cfg_m2 = dict(self.det_cfg_base)
        self.det_cfg_m2['det_path'] = os.path.expanduser(os.getenv('DET_MODEL_M2', os.getenv('DET_MODEL', self.det_cfg_base['det_path'])))
        self.det_cfg_m2['seg_path'] = os.path.expanduser(os.getenv('SEG_MODEL_M2', os.getenv('SEG_MODEL', self.det_cfg_base['seg_path'])))
        self.det_cfg_m2['cls_path'] = os.path.expanduser(os.getenv('CLASS_MODEL_M2', self.det_cfg_base['cls_path']))
        self.det_cfg_m2['expected_color'] = os.getenv('EXPECTED_COLOR_M2', 'purple')


        self.worker1 = None
        self.worker2 = None
        
        self._snap_q = queue.Queue(maxsize=16)
        self._decider = None

        self._decider = DecideWorker(self._snap_q, self.det_cfg_m1, self._on_decision)
        self._decider.start()
        
        self.status_lbl.config(text="대기")

        # Models (요약 패널)
        model_area = ttk.Frame(root); model_area.grid(row=1, column=0, sticky='nsew', padx=(0,6))
        model_area.grid_columnconfigure(0, weight=1); model_area.grid_columnconfigure(1, weight=1)
        self.m1_vars = {k: tk.StringVar(value='-') for k in ['good','sev','par','par_weight','def_weight','overall','rate']}
        self.m2_vars = {k: tk.StringVar(value='-') for k in ['good','sev','par','par_weight','def_weight','overall','rate']}
        self._model_panel_simple(model_area, "Model 1", self.m1_vars).grid(row=0, column=0, sticky='nsew', padx=(0,6))
        self._model_panel_simple(model_area, "Model 2", self.m2_vars).grid(row=0, column=1, sticky='nsew', padx=(6,0))

        # Recent — DB 전용
        recent = ttk.LabelFrame(root, text="최근 50건")
        recent.grid(row=1, column=1, rowspan=2, sticky='nsew', padx=(6,0), pady=(0,0))
        self.tree = ttk.Treeview(
            recent,
            columns=("ts","m1_good","m1_sev","m1_par","m2_good","m2_sev","m2_par"),
            show='headings', height=24
        )
        for col, text, w in [
            ("ts","시간",170),("m1_good","M1 정상",70),("m1_sev","M1 완전 불량",70),("m1_par","M1 부분 불량",70),
            ("m2_good","M2 정상",70),("m2_sev","M2 완전 불량",70),("m2_par","M2 부분 불량",70)
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor='center')
        self.tree.pack(fill='both', expand=True, padx=6, pady=6)

        self._fill_recent50_once()

        # (너의 Treeview 생성 블록 바로 아래)
        self._recent_last_key = None                # ← 마지막으로 본 (id,ts) 키 저장
        self.after(300, self._recent_tick)          # ← 0.3초 뒤부터 주기 갱신 시작
                
        # 장치 패널 — 가로 전체
        self._build_device_panel(root)
        # 처음 화면에 STOP 램프가 반드시 들어오도록
        self._sync_start_stop_lamps()

        # === 액츄에이터 제어용: 디바운스 + 지연(arming) ===
        self._act1_last = 0.0
        self._act2_last = 0.0
        self._act_debounce = float(os.getenv('ACT_DEBOUNCE_SEC', '0.2'))  # 0.8초 디바운스
        self._act_delay = float(os.getenv('ACT_DELAY_SEC', '0.3'))        # 0.5초 지연(감지 후)
        
        # --- [추가] 센서 게이팅용 3초 무장 창 ---
        self._arm1_deadline = 0.0   # CAM1 유효 마감시각 (epoch)
        self._arm2_deadline = 0.0   # CAM2 유효 마감시각 (epoch)
        self._arm1_cond = None      # 'sev' 등
        self._arm2_cond = None      # 'par' or 'good'

        self._prev_s1 = False
        self._prev_s2 = False
        self._pending1 = None
        self._pending2 = None
        self._prev_start_btn = False
        self._prev_stop_btn  = False
        
        # 기존 짧은 엣지 디바운스(남겨둠, 내부 보조용)
        self._btn_debounce = float(os.getenv('BTN_DEBOUNCE_SEC', '0.15'))
        self._last_start_edge = 0.0
        self._last_stop_edge  = 0.0

        # STOP 후 START 잠금(기존 로직)
        self._stop_latched_until = 0.0

        # ★ 3초 레벨 디바운스(눌린 상태 무시)용 타이머
        self._start_db_sec = float(os.getenv('START_DEBOUNCE_SEC', '3.0'))
        self._stop_db_sec  = float(os.getenv('STOP_DEBOUNCE_SEC',  '3.0'))
        self._start_ignore_until = 0.0
        self._stop_ignore_until  = 0.0

        # 실시간 카운터(M1,M2) + DB 이벤트 옵션
        self._counts = {
            1: {"good":0, "sev":0, "par":0},
            2: {"good":0, "sev":0, "par":0},
        }
        self._write_db_on_event = bool(int(os.getenv("WRITE_DB_ON_EVENT", "1")))
        # timers
        self.after(120, self.update_camera)
        self.after(200, self._poll_device_panel)
        self.after(180, self._logic_loop)  # stain% & 센서 연동 로직
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 모델이 없으면 RUN 버튼 비활성화(선택)
        if not _has_model(self.det_cfg_m1):
            self.btn_m1.config(state="disabled", text="Model 1 (미설치)")
        if not _has_model(self.det_cfg_m2):
            self.btn_m2.config(state="disabled", text="Model 2 (미설치)")
            
        self._buzz_arm = {"sensor": None, "pattern": None, "deadline": 0.0}  # 단일 버저 ARM 슬롯

    def destroy(self):
        try:
            if self.worker1: self.worker1.stop()
            if self.worker2: self.worker2.stop()
            if hasattr(self, "_decider") and self._decider: self._decider.stop()   # [추가]
        except Exception: pass
        try: self.cam1.stop(); self.cam2.stop()
        except Exception: pass
        try:
            if hasattr(self, "hw") and self.hw: self.hw.close()
        except Exception: pass
        return super().destroy()

    def _row(self, parent, r, name, var):
        ttk.Label(parent, text=name, width=16).grid(row=r, column=0, sticky='w', padx=4, pady=2)
        ttk.Label(parent, textvariable=var, width=12).grid(row=r, column=1, sticky='w', padx=4, pady=2)

    def _fill_recent50_once(self):
        """
        DB에서 최근 50건을 읽어 Treeview를 '필요할 때만' 갱신한다.
        - 맨 위 (id, ts)가 이전과 같으면 아무 것도 하지 않음 → 깜빡임 최소화
        - 바뀌었을 때만 전체 갱신
        """
        try:
            rows, _ = db_query(f'''
                SELECT id, ts, m1_good, m1_sev, m1_par, m2_good, m2_sev, m2_par
                FROM {DB_TABLE_6}
                ORDER BY ts DESC, id DESC
                LIMIT 50
            ''')
        except Exception:
            rows = []

        # 표시용으로 가공
        recent_rows = [
            (
                ts.strftime('%Y-%m-%d %H:%M:%S'),
                int(m1g or 0), int(m1s or 0), int(m1p or 0),
                int(m2g or 0), int(m2s or 0), int(m2p or 0)
            )
            for _id, ts, m1g, m1s, m1p, m2g, m2s, m2p in rows
        ]

        # 변경 확인용 키: 맨 위 레코드의 (id,ts) 조합
        new_top_key = None
        if rows:
            _id0, ts0 = rows[0][0], rows[0][1]
            new_top_key = f"{_id0}:{ts0.strftime('%Y-%m-%d %H:%M:%S')}"

        # 이전과 동일하면 무변경 → 리턴
        if new_top_key and getattr(self, "_recent_last_key", None) == new_top_key:
            return False

        # 여기서만 전체 리프레시 (변경 있을 때만 실행되므로 깜빡임 최소화)
        try:
            for iid in self.tree.get_children(""):
                self.tree.delete(iid)
        except Exception:
            pass

        for row in recent_rows:
            self.tree.insert("", "end", values=row)

        self._recent_last_key = new_top_key
        return True

    def _recent_tick(self):
        """최근 50건을 필요할 때만 갱신하고, 다음 주기를 예약한다."""
        try:
            self._fill_recent50_once()  # 변경 없으면 내부에서 아무 것도 안 함
        except Exception:
            pass
        # 1초 간격으로 폴링 (원하면 1500~2000ms로 늘려도 됨)
        self.after(1000, self._recent_tick)

    def _model_panel_simple(self, parent, title, vars_dict):
        lf = ttk.LabelFrame(parent, text=title)
        g = ttk.Frame(lf); g.pack(fill='x', padx=8, pady=8)
        items=[("정상품","good"),("완전오염","sev"),("30%오염","par"),
               ("Partial 가중","par_weight"),("불량(가중)","def_weight"),
               ("총 처리","overall"),("가중 불량률","rate")]
        for i,(nm,key) in enumerate(items):
            ttk.Label(g, text=nm, width=14).grid(row=i, column=0, sticky='w', padx=4, pady=2)
            ttk.Label(g, textvariable=vars_dict[key], width=10).grid(row=i, column=1, sticky='w', padx=4, pady=2)
        return lf

    # === RUN/STOP ===
    def on_run(self, model_id:int):
        err = None
        # arming 전에 RUN 금지 (초기 깜빡임/오동작 방지)
        if not self._armed:
            self.status_lbl.config(text="초기화 중(잠시 후 RUN 가능)")
            return

        # 모델 가드
        if model_id == 1 and not _has_model(self.det_cfg_m1):
            messagebox.showwarning("Model 1", "Model 1 파일이 없습니다.\n(DET_MODEL_M1 / SEG_MODEL_M1)")
            return
        if model_id == 2 and not _has_model(self.det_cfg_m2):
            messagebox.showwarning("Model 2", "Model 2 파일이 없습니다.\n(DET_MODEL_M2 / SEG_MODEL_M2)")
            return
        if not self._restart_workers_for(model_id):
            return
        
        self._is_running = True
        self.active_model = model_id
        self._sync_start_stop_lamps()
        
        self._set_beacons(red=False, orange=False, green=False)
        
        self._pattern_on_model_start(model_id)

        self.status_lbl.config(text=f"Model {model_id} 작동중")
        if model_id == 1:
            self.m1_state.set("작동중"); self.m2_state.set("대기")
        else:
            self.m1_state.set("대기"); self.m2_state.set("작동중")

        # 백엔드 알림 비동기
        try:
            url = os.getenv("BACKEND_URL")
            if url:
                def _notify_run(mid):
                    import json, urllib.request
                    req = urllib.request.Request(url.rstrip("/")+"/control",
                        data=json.dumps({"action":"run","model":mid}).encode("utf-8"),
                        headers={"Content-Type":"application/json"})
                    urllib.request.urlopen(req, timeout=1.5)
                self._bg(_notify_run, model_id)
        except Exception as e:
            err = err or e

        # Conveyor ON 확정
        try:
            if self.hw: self.hw.set_relay(6, True)  # Conveyor ON
            if hasattr(self, "_relay_vars") and 6 in self._relay_vars:
                self._relay_vars[6].set(True)
                self._refresh_relay_visual(6)
        except Exception:
            pass

        if err:
            messagebox.showwarning("제어 알림", f"장치/백엔드 알림 중 오류: {err}")

    def _stop_workers(self):
        # 두 워커 정지 및 해제
        for w in (self.worker1, self.worker2):
            try:
                if w: w.stop()
            except Exception:
                pass
        for w in (self.worker1, self.worker2):
            try:
                if w:w.join(timeout=0.8) 
            except Exception:
                pass
        self.worker1 = None
        self.worker2 = None

    def _start_workers_with_cfg(self, cfg):
        # 두 카메라 모두 같은 cfg(선택한 모델)로 추론 워커 시작
        if cv2 is None or not PIL_OK:
            self.status_lbl.config(text="OpenCV/Pillow 필요"); 
            return False
        try:
            self.worker1 = DetectionWorker(self.cam1.get_frame, dict(cfg), self._snap_q)
            self.worker1.start()
            self.worker2 = DetectionWorker(self.cam2.get_frame, dict(cfg), self._snap_q)
            self.worker2.start()

            # [추가] 판정 워커 시작 (중복 방지)
            try:
                if self._decider:
                    self._decider.stop()
                    self._decider = None
            except Exception:
                pass
            self._decider = DecideWorker(self._snap_q, dict(cfg), self._on_decision)
            self._decider.start()
            
            self.status_lbl.config(text="Det/Seg 가동(오버레이 표시) · CAM1/CAM2=같은 모델")
            return True
        except Exception as e:
            self.status_lbl.config(text=f"Worker 시작 실패: {e}")
            self._stop_workers()
            return False

    def _restart_workers_for(self, model_id:int):
        # model_id=1이면 det_cfg_m1, 2면 det_cfg_m2를 두 카메라에 동시에 적용
        cfg = self.det_cfg_m1 if model_id == 1 else self.det_cfg_m2
        if not _has_model(cfg):
            messagebox.showwarning(
                f"Model {model_id}",
                f"Model {model_id} 파일이 없습니다.\n(DET_MODEL_M{model_id} / SEG_MODEL_M{model_id})"
            )
            return False
        self._stop_workers()
        return self._start_workers_with_cfg(cfg)



    def on_stop(self):
        err = None
        self.active_model = 0
        self._is_running = False
        self._stop_workers()
        self._sync_start_stop_lamps()
        self.status_lbl.config(text="대기")
        self.m1_state.set("대기"); self.m2_state.set("대기")

        # 백엔드 알림 비동기
        try:
            url = os.getenv("BACKEND_URL")
            if url:
                def _notify_stop():
                    import json, urllib.request
                    req = urllib.request.Request(url.rstrip("/")+"/control",
                        data=json.dumps({"action":"stop"}).encode("utf-8"),
                        headers={"Content-Type":"application/json"})
                    urllib.request.urlopen(req, timeout=1.5)
                self._bg(_notify_stop)
        except Exception as e:
            err = err or e

        # Conveyor OFF 확정
        try:
            if self.hw: self.hw.set_relay(6, False)  # Conveyor OFF
            if hasattr(self, "_relay_vars") and 6 in self._relay_vars:
                self._relay_vars[6].set(False)
                self._refresh_relay_visual(6)
        except Exception:
            pass

        if err:
            messagebox.showwarning("제어 알림", f"장치/백엔드 알림 중 오류: {err}")

    # === Camera update (dual, overlay 우선) ===
    def update_camera(self):
        try:
            if cv2 is None or not PIL_OK:
                self.cam1_lbl.configure(text="(OpenCV/Pillow 필요)")
                self.cam2_lbl.configure(text="(OpenCV/Pillow 필요)")
            else:
                resample = dict(LANCZOS=Image.LANCZOS, BILINEAR=Image.BILINEAR, BICUBIC=Image.BICUBIC).get(RESAMPLE.upper(), Image.LANCZOS)

                # CAM1: worker 오버레이가 있으면 우선 사용
                f1 = self.worker1.last_annot if (self.worker1 and self.worker1.last_annot is not None) else self.cam1.get_frame()
                if f1 is not None:
                    rgb1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                    img1 = Image.fromarray(rgb1).resize((CAM_VIEW_W, CAM_VIEW_H), resample=resample).copy()
                    self._tk_img1 = ImageTk.PhotoImage(img1)
                    self.cam1_lbl.configure(image=self._tk_img1, text='')

                # CAM2: worker 오버레이 우선
                f2 = self.worker2.last_annot if (self.worker2 and self.worker2.last_annot is not None) else self.cam2.get_frame()
                if f2 is not None:
                    rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    img2 = Image.fromarray(rgb2).resize((CAM2_VIEW_W, CAM2_VIEW_H), resample=resample).copy()
                    self._tk_img2 = ImageTk.PhotoImage(img2)
                    self.cam2_lbl.configure(image=self._tk_img2, text='')
        except Exception as e:
            self.cam1_lbl.configure(text=f"(CAM1 오류: {e})")
            self.cam2_lbl.configure(text=f"(CAM2 오류: {e})")
        finally:
            self.after(90, self.update_camera)

    # === 백그라운드 실행 헬퍼 ===
    def _bg(self, target, *args, **kwargs):
        th = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        th.start()
        return th

    # === 공통 갱신 로직(중복 제거) ===
    def _apply_simple_node(self, node, vars_dict):
        sev = int(node['sev']); par = int(node['par']); good = int(node['good'])
        par_weight = round(par * PARTIAL_WEIGHT, 2)
        def_weight = round(sev + par_weight, 2)
        overall = good + sev + par
        rate = round((def_weight/overall*100.0), 1) if overall>0 else 0.0
        vars_dict['good'].set(good)
        vars_dict['sev'].set(sev)
        vars_dict['par'].set(par)
        vars_dict['par_weight'].set(f"{par_weight:.2f}")
        vars_dict['def_weight'].set(f"{def_weight:.2f}")
        vars_dict['overall'].set(overall)
        vars_dict['rate'].set(f"{rate}%")
        return overall, def_weight, good

    def _set_kpi(self, total, good, defw, rate):
        self.var_total.set(total); self.var_good.set(good)
        self.var_defw.set(f"{defw:.2f}"); self.var_rate.set(f"{rate}%")
        self.pb['value'] = min(100, rate)

    def _update_from_counts(self, m1_counts: dict, m2_counts: dict):
        m1_overall, m1_defw, m1_good = self._apply_simple_node(m1_counts, self.m1_vars)
        m2_overall, m2_defw, m2_good = self._apply_simple_node(m2_counts, self.m2_vars)
        total = m1_overall + m2_overall
        total_good = m1_good + m2_good
        total_defw = round(m1_defw + m2_defw, 2)
        rate = round((total_defw/total*100.0), 1) if total>0 else 0.0
        self._set_kpi(total, total_good, total_defw, rate)
        self._maybe_blink_beacon(rate)
        # KPI 확장 항목
        self.k_defw_abs.set(f"{total_defw:.2f}")
        win_minutes = max(1e-9, parse_window(self.win_var.get()).total_seconds()/60.0)
        tput = total / win_minutes if total > 0 else 0.0
        self.k_tput.set(f"{tput:.1f} 개/분")

    def _win_minutes(self) -> int:
        s = (self.win_var.get() or '5m').strip().lower()
        if s.endswith('m'):
            return max(1, int(s[:-1] or 5))
        if s.endswith('h'):
            return max(1, int(s[:-1] or 1) * 60)
        if s.endswith('d'):
            return max(1, int(s[:-1] or 1) * 1440)
        return 5

    def _set_lamp(self, canvas: tk.Canvas, on: bool):
        if not canvas:
            return
        canvas.delete("all")
        color = "#2ecc71" if on else "#95a5a6"
        canvas.create_oval(2, 2, 20, 20, fill=color, outline="black")

    def _sync_start_stop_lamps(self):
        """START/STOP 타일(왼쪽 하단)의 램프를 시스템 상태(_is_running) 기준으로 표시"""
        try:
            # START 버튼 타일 = RUN일 때 초록, STOP 버튼 타일 = RUN 아닐 때 초록
            if hasattr(self, "_sensor_lamps"):
                self._set_lamp(self._sensor_lamps.get("start_button"), bool(self._is_running))
                self._set_lamp(self._sensor_lamps.get("stop_button"),  not bool(self._is_running))
        except Exception:
            pass

    # ==== Beacon helpers ==================================================
    def _set_beacons(self, red=None, orange=None, green=None):
        mapping = {1: red, 2: orange, 3: green}
        for idx, val in mapping.items():
            if val is None:
                continue
            try:
                self._relay_vars[idx].set(bool(val))
                if self.hw:
                    self.hw.set_relay(idx, bool(val))
                self._refresh_relay_visual(idx)
            except Exception:
                pass
            
    def _pulse_beacon(self, kind: str, ms: int = None):
        """
        kind: "good" | "par" | "sev"
        요청 패턴:
        - sev(RED):   0.5s ON → 0.5s OFF → 0.5s ON → 0.5s OFF  (총 2.0s)
        - par(ORANGE):0.5s ON → 0.5s OFF                       (총 1.0s)
        - good(GREEN):1.0s ON                                  (총 1.0s)
        """
        if getattr(self, "_beacon_busy", False):
            return
        # 먼저 모두 OFF
        try:
            self._set_beacons(red=False, orange=False, green=False)
        except Exception:
            pass

        def seq_red():
            # ON 0.5s
            self._set_beacons(red=True)
            self.after(500, lambda: (
                self._set_beacons(red=False),
                self.after(500, lambda: (
                    self._set_beacons(red=True),
                    self.after(500, lambda: self._set_beacons(red=False))
                ))
            ))

        def seq_orange():
            self._set_beacons(orange=True)
            self.after(500, lambda: self._set_beacons(orange=False))  # 0.5s 후 OFF (총 1.0s 간격)

        def seq_green():
            self._set_beacons(green=True)
            self.after(1000, lambda: self._set_beacons(green=False))

        if kind == "sev":
            seq_red()
        elif kind == "par":
            seq_orange()
        elif kind == "good":
            seq_green()
        else:
            return

    # === 모델 시작 패턴: 비프 + GREEN 점멸 ===============================
    def _beep(self, n=1, dur=0.1, gap=0.1):
        """릴레이 #4 (Buzzer) 짧게 n회 비프. GUI 블로킹 방지: after 체인."""
        if not self.hw:
            return
        total_steps = n * 2  # ON/OFF 세트
        def step(i=0):
            try:
                on = (i % 2 == 0)
                self.hw.set_relay(4, on)
                if 4 in self._relay_vars:
                    self._relay_vars[4].set(on); self._refresh_relay_visual(4)
            except Exception:
                pass
            if i + 1 < total_steps:
                self.after(int(1000*(dur if on else gap)), lambda: step(i+1))
            else:
                # 마지막은 확실히 OFF
                try:
                    self.hw.set_relay(4, False)
                    if 4 in self._relay_vars:
                        self._relay_vars[4].set(False); self._refresh_relay_visual(4)
                except Exception:
                    pass
        step(0)

    def _blink_green(self, count=1, on_ms=500, off_ms=200):
        """릴레이 #3 (Green Beacon) 점멸 count회."""
        if not self.hw:
            return
        def one_blink(k):
            # ON
            try:
                self._set_beacons(green=True)
            except Exception:
                pass
            # OFF 예약
            self.after(on_ms, lambda: (
                self._set_beacons(green=False),
                self.after(off_ms, lambda: one_blink(k-1) if k-1>0 else None)
            ))
        # 먼저 모두 OFF
        try:
            self._set_beacons(red=False, orange=False, green=False)
        except Exception:
            pass
        one_blink(count)

    def _pattern_on_model_start(self, model_id:int):
        """Model 1: 비프 1회 + 초록 0.5s 1회 / Model 2: 비프 2회 + 초록 0.5s 2회"""
        if model_id == 1:
            self._beep(n=1, dur=0.1, gap=0.1)
            self._blink_green(count=1, on_ms=500, off_ms=200)
        else:
            self._beep(n=2, dur=0.1, gap=0.1)
            self._blink_green(count=2, on_ms=500, off_ms=200)

    def _maybe_blink_beacon(self, rate: float, on_ms: int = 800):
        # 부팅 시퀀스(혹은 다른 배타적 제어) 중엔 개입 금지
        if getattr(self, "_beacon_busy", False):
            return
        desired = 'none'
        if rate >= 100.0:
            desired = 'red'
        elif rate >= 30.0:
            desired = 'orange'
        elif rate == 0.0:
            desired = 'green'

        if desired == self._last_beacon:
            return

        self._last_beacon = desired
        if self._beacon_after:
            try: self.after_cancel(self._beacon_after)
            except Exception: pass
            self._beacon_after = None

        self._set_beacons(red=False, orange=False, green=False)
        if desired == 'none':
            return
        if desired == 'red':
            self._set_beacons(red=True)
        elif desired == 'orange':
            self._set_beacons(orange=True)
        elif desired == 'green':
            self._set_beacons(green=True)
        try:
            self._beacon_after = self.after(on_ms, lambda: self._set_beacons(red=False, orange=False, green=False))
        except Exception:
            pass

        

    def _beacon_boot_chase(self, step_ms: int = 250, hold_ms: int = 1000):
        """
        앱 시작시: RED -> ORANGE -> GREEN 순서로 "누적" 켠 뒤,
        GREEN까지 켜진 상태로 hold_ms 유지하고 모두 OFF.
        """
        # === 잠금 시작 ===
        self._beacon_busy = True

        # 기존에 깜빡임 예약이 걸려 있었다면 취소 (중복 개입 차단)
        try:
            if getattr(self, "_beacon_after", None):
                self.after_cancel(self._beacon_after)
                self._beacon_after = None
        except Exception:
            pass

        try:
            # 모두 끈 상태에서 시작
            self._set_beacons(red=False, orange=False, green=False)
        except Exception:
            pass

        def on_red():
            # RED 켜고 유지
            self._set_beacons(red=True, orange=False, green=False)
            self.after(step_ms, on_orange)

        def on_orange():
            # ORANGE만 True 전달(RED는 None이라 그대로 유지)
            self._set_beacons(red=True, orange=True, green=False)
            self.after(step_ms, on_green)

        def on_green():
            # GREEN만 True 전달(RED/ORANGE는 None이라 그대로 유지)
            self._set_beacons(red=True, orange=True, green=True)
            # 모든 불이 켜진 상태를 hold_ms 유지
            self.after(hold_ms, all_off)

        def all_off():
            self._set_beacons(red=False, orange=False, green=False)
            # === 잠금 해제 ===
            self._beacon_busy = False
        # 살짝 딜레이 후 시작
        self.after(50, on_red)


    # ======================================================================
    # App 클래스 안에 추가 (기존 _set_recent가 있다면 이 버전으로 교체)
    def _set_recent(self, recent_rows):
        """
        recent_rows: 최신이 앞쪽에 오는 리스트.
        트리에 데이터가 없을 때만 전체 채우고,
        그 외에는 깜빡임 없이 차분 적용(_set_recent_diff)으로 위임.
        """
        try:
            children = self.tree.get_children("")
        except Exception:
            children = ()
        if not children:
            # 최초 1회만 풀 렌더 (깜빡임 거의 없음)
            for row in recent_rows:
                self.tree.insert("", "end", values=row)
            return
        # 이후부터는 차분 적용
        self._set_recent_diff(recent_rows)

    def _update_row_if_changed(self, iid, new_values):
        """iid 행의 값이 new_values와 다르면 그 행만 갱신(깜빡임 없음)."""
        try:
            cur = self.tree.item(iid, "values")
            # values는 튜플/리스트 문자열들이라 문자열 비교로 충분
            if tuple(map(str, cur)) != tuple(map(str, new_values)):
                self.tree.item(iid, values=new_values)
        except Exception:
            pass

    def _set_recent_diff(self, recent_rows):
        """
        깜빡임 없이 신규만 위로 추가, 기존 동일 타임스탬프는 값만 갱신.
        최신이 앞쪽(0번)으로 온다는 가정.
        """
        try:
            children = list(self.tree.get_children(""))
        except Exception:
            children = []

        # 현재 트리 최상단 ts
        top_ts = None
        if children:
            try:
                top_ts = self.tree.item(children[0], "values")[0]
            except Exception:
                top_ts = None

        # 1) 위쪽으로 새로 추가할 개수 계산
        #    recent_rows는 최신이 앞쪽이므로, 기존 top_ts를 만날 때까지가 "신규"
        new_count = 0
        for row in recent_rows:
            ts = row[0]
            if top_ts is not None and str(ts) == str(top_ts):
                break
            new_count += 1

        # 2) 신규를 역순으로 맨 위에 insert (시간 오름차순 유지 위해 역순)
        if new_count > 0:
            for row in reversed(recent_rows[:new_count]):
                self.tree.insert("", 0, values=row)

        # 3) 겹치는 구간(=이미 트리에 있는 범위)은 값이 바뀌었으면 그 행만 갱신
        #    recent_rows[new_count:] 를 트리 상단부터 차례로 대응시켜 비교 갱신
        start = new_count
        iids = list(self.tree.get_children(""))
        for idx, row in enumerate(recent_rows[start:], start=0):
            if idx >= len(iids):
                break
            self._update_row_if_changed(iids[idx], row)

        # 4) 행 수 50개로 유지 (초과분은 아래쪽부터 삭제)
        MAX_N = 50
        iids = list(self.tree.get_children(""))
        if len(iids) > MAX_N:
            for iid in iids[MAX_N:]:
                self.tree.delete(iid)

    # === 장치 패널(임베디드) ===
    def _build_device_panel(self, root):
        panel = ttk.LabelFrame(root, text=f"장치 패널 — {'device panel' if self.hw.is_dummy else self.hw.dev_name}")
        panel.grid(row=2, column=0, columnspan=1, sticky='nsew', padx=(0,6), pady=(6,0))

        grid = ttk.Frame(panel); grid.pack(fill='both', expand=True, padx=6, pady=6)
        for r in range(3):
            grid.rowconfigure(r, weight=1, uniform="devrow")
        for c in range(4):
            grid.columnconfigure(c, weight=1, uniform="devcol")

        self._relay_vars = {}
        self._relay_lamps = {}
        names = ["Beacon Red","Beacon Orange","Beacon Green","Buzzer","LED","Conveyor","Actuator 1","Actuator 2"]

        for i in range(8):
            r = i // 4
            c = i % 4
            fr = ttk.Labelframe(grid, text=names[i], padding=8)
            fr.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            var = tk.BooleanVar(value=False)
            self._relay_vars[i+1] = var
            lamp = tk.Canvas(fr, width=22, height=22, highlightthickness=0)
            lamp.grid(row=0, column=0, padx=4, pady=4)
            self._relay_lamps[i+1] = lamp
            self._set_lamp(lamp, False)
            fr.columnconfigure(0, weight=1)

        self._sensor_lamps = {}
        sensor_titles = [
            ("START BUTTON", "start_button"),
            ("STOP BUTTON",  "stop_button"),
            ("SENSOR 1",     "sensor_1"),
            ("SENSOR 2",     "sensor_2"),
        ]
        for c, (title, key) in enumerate(sensor_titles):
            fr = ttk.Labelframe(grid, text=title, padding=8)
            fr.grid(row=2, column=c, padx=4, pady=4, sticky="nsew")
            lamp = tk.Canvas(fr, width=22, height=22, highlightthickness=0)
            lamp.grid(row=0, column=0, padx=4, pady=4)
            self._sensor_lamps[key] = lamp
            self._set_lamp(lamp, False)
            fr.columnconfigure(0, weight=1)

    def _refresh_relay_visual(self, idx:int):
        on = self._relay_vars[idx].get()
        lamp = self._relay_lamps.get(idx)
        self._set_lamp(lamp, on)

    def _poll_device_panel(self):
        try:
            st = self.hw.get_sensors() if self.hw else {
                "start_button": False, "stop_button": False, "sensor_1": False, "sensor_2": False
            }
        except Exception:
            st = {"start_button": False, "stop_button": False, "sensor_1": False, "sensor_2": False}

        try:
            outs = self.hw.get_outputs() if self.hw else {}
        except Exception:
            outs = {}

        # 램프 상태 갱신
        try:
            for idx, on in outs.items():
                var = self._relay_vars.get(idx)
                if var:
                    if var.get() != bool(on):
                        var.set(bool(on))
                    self._refresh_relay_visual(idx)
        except Exception:
            pass

        # === 하드웨어 START/STOP 버튼 처리 (3초 레벨 디바운스 + STOP 우선) ===
        try:
            b_start = bool(st.get("start_button", False))
            b_stop  = bool(st.get("stop_button",  False))
            now = time.time()

            # ---- STOP: Rising-edge + 3초 레벨 디바운스 ----
            if b_stop and not self._prev_stop_btn:
                if now >= self._stop_ignore_until:
                    self._stop_ignore_until = now + self._stop_db_sec     # 3초간 추가 입력 무시
                    self._last_stop_edge = now
                    self._stop_latched_until = now + 1.0                  # (기존) 1초 START 락
                    # UI STOP 루틴 호출 (워커/컨베이어 정지 포함)
                    self.on_stop()
                    # 안전하게 비콘/컨베이어 확실히 OFF
                    try:
                        self._set_beacons(red=False, orange=False, green=False)
                    except Exception: pass
                    try:
                        if self.hw:
                            self.hw.set_relay(6, False)  # Conveyor OFF
                            if hasattr(self.hw, "system_stop"):
                                self.hw.system_stop()
                        if hasattr(self, "_relay_vars") and 6 in self._relay_vars:
                            self._relay_vars[6].set(False)
                            self._refresh_relay_visual(6)
                    except Exception: pass

            # ---- START: Rising-edge + 3초 레벨 디바운스 + 모델 토글 ----
            if b_start and not self._prev_start_btn:
                if now >= self._start_ignore_until:
                    self._start_ignore_until = now + self._start_db_sec   # 3초간 추가 입력 무시
                    self._last_start_edge = now
                    # STOP과 동시가 아니고, STOP 락 해제되었고, arming 완료만 확인
                    if self._armed and (not b_stop) and (now >= self._stop_latched_until):
                        if not self._is_running or self.active_model == 0:
                            # 처음 시작 → Model 1
                            self.on_run(1)
                        else:
                            # 토글: 1 ↔ 2
                            next_model = 2 if self.active_model == 1 else 1
                            self.on_run(next_model)

            # 이전 상태 저장 (엣지 검출용)
            self._prev_start_btn = b_start
            self._prev_stop_btn  = b_stop
        except Exception:
            pass

        # 센서 램프
        try:
            self._sync_start_stop_lamps()  # START/STOP은 시스템 상태 기준
            if "sensor_1" in self._sensor_lamps:
                self._set_lamp(self._sensor_lamps["sensor_1"], bool(st.get("sensor_1", False)))
            if "sensor_2" in self._sensor_lamps:
                self._set_lamp(self._sensor_lamps["sensor_2"], bool(st.get("sensor_2", False)))

        except Exception:
            pass

        self.after(200, self._poll_device_panel)

    # === Stain% & 센서 연동 로직 (카메라 역할 고정 + 디바운스/지연 + 엣지 스냅샷) ===
    def _logic_loop(self):
        """
        센서 라이징 엣지에서 stain% 스냅샷을 저장하고 0.5초 뒤 액츄에이터 펄스.
        - CAM1: r1 >= 31% AND sensor_1 rising → 0.5s 뒤 릴레이7 (kind='sev')
        - CAM2: 1% <= r2 <= 30% AND sensor_2 rising → 0.5s 뒤 릴레이8 (kind='par')
        0%는 동작 없음. 디바운스 적용.
        """
        try:
            st = self.hw.get_sensors() if self.hw else {"sensor_1":False,"sensor_2":False}
            s1 = bool(st.get("sensor_1", False))
            s2 = bool(st.get("sensor_2", False))
            now = time.time()

            # 실시간 stain% 표시(있으면)
            r1 = self.worker1.last_ratio if self.worker1 else None
            r2 = self.worker2.last_ratio if self.worker2 else None
            if r1 is not None: self.r1_var.set(f"CAM1: {r1:.1f}%")
            if r2 is not None: self.r2_var.set(f"CAM2: {r2:.1f}%")

            # --- SENSOR 1: CAM1 = 완전오염 전용 ---
            # --- SENSOR 1: CAM1 (무장창 내에서만 유효) ---
            if s1 and not self._prev_s1 and self._is_running:
                now = time.time()
                if (self._arm1_cond == "sev") and (now <= self._arm1_deadline):
                    if (now - self._act1_last) >= self._act_debounce and self._pending1 is None:
                        self._pending1 = {"when": now, "kind": "sev"}
                        delay_ms = int(self._act_delay * 1000)  # 0.3s
                        self.after(delay_ms, self._do_act1)
                # 트리거 이후 즉시 해제(중복 방지)
                self._arm1_deadline = 0.0
                self._arm1_cond = None

            # --- SENSOR 2: CAM2 (무장창 내에서만 유효) ---
            if s2 and not self._prev_s2 and self._is_running:
                now = time.time()
                if (self._arm2_cond in ("par", "good")) and (now <= self._arm2_deadline):
                    # par: Act2 + 주황 패턴 / good: 액츄에이터 없이 green 패턴만
                    if self._arm2_cond == "par":
                        if (now - self._act2_last) >= self._act_debounce and self._pending2 is None:
                            self._pending2 = {"when": now, "kind": "par"}
                            delay_ms = int(self._act_delay * 1000)  # 0.3s
                            self.after(delay_ms, self._do_act2)
                    elif self._arm2_cond == "good":
                        # 0.3s 지연 후 GREEN 패턴만 1회
                        def _green_only():
                            # DB 카운터(정상품) + 비콘 그린 패턴
                            self._record_event(kind="good")
                        self.after(int(self._act_delay * 1000), _green_only)
                # 트리거 이후 즉시 해제(중복 방지)
                self._arm2_deadline = 0.0
                self._arm2_cond = None

            # 이전 상태 저장
            self._prev_s1 = s1
            self._prev_s2 = s2

        except Exception:
            pass
        finally:
            self.after(60, self._logic_loop)  # 주기 60ms

    def _do_act1(self):
        info = self._pending1; self._pending1 = None
        self._act1_last = time.time()
        self._pulse_actuator(7, ms=int(os.getenv('ACT1_PULSE_MS', '250')))
        self._bg(self._beep_pattern, "sev")
        if info:
            self._record_event(kind=info.get("kind","sev"))

    def _do_act2(self):
        info = self._pending2; self._pending2 = None
        self._act2_last = time.time()
        self._pulse_actuator(8, ms=int(os.getenv('ACT2_PULSE_MS', '250')))
        self._bg(self._beep_pattern, "par")
        if info:
            self._record_event(kind=info.get("kind","par"))

    def _pulse_actuator(self, relay_idx:int, ms:int=250):
        """릴레이 idx를 ms 밀리초 동안 ON 후 자동 OFF"""
        try:
            self._relay_vars[relay_idx].set(True)
            if self.hw: self.hw.set_relay(relay_idx, True)
            self._refresh_relay_visual(relay_idx)
            self.after(ms, lambda: self._pulse_off(relay_idx))
        except Exception:
            pass

    def _pulse_off(self, relay_idx:int):
        try:
            self._relay_vars[relay_idx].set(False)
            if self.hw: self.hw.set_relay(relay_idx, False)
            self._refresh_relay_visual(relay_idx)
        except Exception:
            pass

    def _record_event(self, model:int=None, kind:str="good"):
        # model 미지정 → 현재 활성 모델로 귀속
        if model is None:
            model = self.active_model
        if model not in (1,2):
            return  # STOP 상태면 기록하지 않음

        node = self._counts[model]
        if kind == "sev":
            node["sev"] += 1
        elif kind == "par":
            node["par"] += 1
        else:
            node["good"] += 1

        # UI 즉시 반영
        self._update_from_counts(self._counts[1], self._counts[2])
                # ★ 이벤트에 맞춰 비콘 펄스
        try:
            self._pulse_beacon(kind)
        except Exception:
            pass


        # 선택적 DB 한 줄 INSERT (WRITE_DB_ON_EVENT=1일 때)
        if self._write_db_on_event and pymysql is not None:
            try:
                cols = ("m1_good","m1_sev","m1_par","m2_good","m2_sev","m2_par")
                vals = [0,0,0,0,0,0]
                idx_map = {
                    (1,"good"): 0, (1,"sev"): 1, (1,"par"): 2,
                    (2,"good"): 3, (2,"sev"): 4, (2,"par"): 5,
                }
                j = idx_map.get((model,kind), None)
                if j is not None:
                    vals[j] = 1
                    sql = f"INSERT INTO {DB_TABLE_6} ({','.join(cols)}) VALUES (%s,%s,%s,%s,%s,%s)"
                    self._bg(lambda: db_query(sql, vals))
            except Exception:
                pass

    def _toggle_admin_panel(self):
        if self._admin_win and self._admin_win.winfo_exists():
            self._admin_win.destroy()
            self._admin_win = None
            self._admin_visible = False
            return
        self._admin_visible = True
        self._build_admin_panel()

    def _build_admin_panel(self):
        win = tk.Toplevel(self)
        self._admin_win = win
        win.title("관리자 패널")
        win.geometry("680x320")
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass

        def make_toggle_cell(parent, title, relay_idx):
            cell = ttk.LabelFrame(parent, text=title, padding=8)
            state = {"on": False}
            def apply(new_state):
                state["on"] = bool(new_state)
                try:
                    if self.hw:
                        self.hw.set_relay(relay_idx, state["on"])
                except Exception:
                    pass
                btn.config(text=("ON" if state["on"] else "OFF"))

                if relay_idx == 6:
                    self._is_running = state["on"]
                    self._sync_start_stop_lamps()
                if relay_idx in self._relay_vars:
                    self._relay_vars[relay_idx].set(state["on"])
                    self._refresh_relay_visual(relay_idx)

            btn = ttk.Button(cell, text="OFF", width=10, command=lambda: apply(not state["on"]))
            btn.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            try:
                outs = self.hw.get_outputs() if self.hw else {}
                cur = bool(outs.get(relay_idx, False))
                apply(cur)
            except Exception:
                pass
            cell.rowconfigure(0, weight=1)
            cell.columnconfigure(0, weight=1)
            return cell

        body = ttk.Frame(win); body.pack(fill="both", expand=True, padx=8, pady=8)
        rows, cols = 2, 4
        for r in range(rows):
            body.rowconfigure(r, weight=1, uniform="admin_row")
        for c in range(cols):
            body.columnconfigure(c, weight=1, uniform="admin_col")

        items = [
            ("Beacon RED", 1),
            ("Beacon ORANGE", 2),
            ("Beacon GREEN", 3),
            ("Buzzer", 4),
            ("LED", 5),
            ("Conveyor", 6),
            ("Actuator 1", 7),
            ("Actuator 2", 8),
        ]
        for i, (title, idx) in enumerate(items):
            r, c = divmod(i, cols)
            cell = make_toggle_cell(body, title, idx)
            cell.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

        def on_close():
            self._admin_visible = False
            win.destroy()
            self._admin_win = None
        win.protocol("WM_DELETE_WINDOW", on_close)

    def _on_close(self):
        self.destroy()
        
    def _buzzer_set(self, on: bool):
        # UI 패널 기준 릴레이 #4가 버저
        try:
            if getattr(self, "hw", None):
                self.hw.set_relay(4, bool(on))
        except Exception:
            pass

    def _beep_pattern(self, pattern: str):
        # 릴레이 특성 고려: 짧게 '딸깍' 수준
        import time
        def pulse(dt=0.1):
            self._buzzer_set(True)
            time.sleep(max(0.01, dt))
            self._buzzer_set(False)

        if pattern == "sev":
            # 0.1s → (0.5s 휴지) → 0.1s → (0.5s 휴지)
            pulse(0.1); time.sleep(0.5); pulse(0.1); time.sleep(0.5)
        elif pattern == "par":
            # 0.1s → (0.5s 휴지)
            pulse(0.1); time.sleep(0.5)

    def _maybe_beep(self, sensor_id: int):
        now = time.time()
        arm = self._buzz_arm
        if arm["sensor"] == sensor_id and now <= arm["deadline"] and arm["pattern"]:
            # 비동기로 울려서 흐름 안막히게
            self._bg(self._beep_pattern, arm["pattern"])
            # 1회 사용 후 소진
            self._buzz_arm = {"sensor": None, "pattern": None, "deadline": 0.0}

    def _on_decision(self, grade, ratio, has_sticker):
        print(f"[DECISION] {grade}, stain={ratio:.1f}%, sticker={has_sticker}")
        now = time.time()

        # NEW: DecideWorker가 준 grade 로 액추에이터 제어 (색상 불일치 포함)
        if grade in ("NG80", "NG_STICKER"):
            # 완전불량 → Actuator 1 (RED)
            self._arm1_deadline = now + 3.0
            self._arm1_cond = "sev"
            self._buzz_arm = {"sensor": 1, "pattern": "sev", "deadline": now + 3.0}

        elif grade == "NG30":
            # 부분불량 → Actuator 2 (ORANGE)
            self._arm2_deadline = now + 3.0
            self._arm2_cond = "par"
            self._buzz_arm = {"sensor": 2, "pattern": "par", "deadline": now + 3.0}

        else:  # "OK"
            # 정상품 → GREEN만 (버저 ARM 없음)
            self._arm2_deadline = now + 3.0
            self._arm2_cond = "good"
            self._buzz_arm = {"sensor": None, "pattern": None, "deadline": 0.0}


# === Camera controls (3초 지연) ===
def delayed_camera_setup():
    time.sleep(3)
    cmds = [
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "auto_exposure=1", "-c", "exposure_dynamic_framerate=0",
         "-c", "white_balance_automatic=0", "-c", "focus_automatic_continuous=0"],
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_time_absolute=60", "-c", "gain=128",
         "-c", "white_balance_temperature=4500", "-c", "focus_absolute=70"],
        ["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1", "-c", "exposure_dynamic_framerate=0",
         "-c", "white_balance_automatic=0", "-c", "focus_automatic_continuous=0"],
        ["v4l2-ctl", "-d", "/dev/video2", "-c", "exposure_time_absolute=60", "-c", "gain=128",
         "-c", "white_balance_temperature=4500", "-c", "focus_absolute=70"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=False)

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # GUI 시작
    app = App()

    # ✅ 3초 후 백그라운드로 카메라 세팅
    threading.Thread(target=delayed_camera_setup, daemon=True).start()

    app.mainloop()