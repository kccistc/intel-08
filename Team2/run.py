#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Intel Geti 스타일: Detection -> ROI Segmentation (OpenVINO) + FactoryController 결합
- 시작 시 컨베이어만 True, 나머지(red/orange/green/led)는 False
- 카메라 프리뷰 + 디텍션 박스 + ROI 세그 오버레이(vivid)
- 다양한 Detection 출력 형식(SSD / YOLOX / 5+cls / 임의 NxK)에 자동 대응
"""

import argparse, time, warnings
import cv2, numpy as np

# OpenVINO
from openvino.runtime import Core

# 프로젝트 하드웨어 컨트롤러
from iotdemo import FactoryController

# ---------- UI ----------
PINK  = (203, 80, 165)   # bbox color
WHITE = (255,255,255)

def letterbox(im, size, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(size/h, size/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=im.dtype)
    top  = (size-nh)//2; left = (size-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def put_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

def _rounded_rect(img, pt1, pt2, color, thickness=2, r=8):
    x1,y1 = pt1; x2,y2 = pt2
    if r<=0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
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

def draw_box(frame, x1,y1,x2,y2, score, cls, labels, color=PINK):
    _rounded_rect(frame, (x1,y1), (x2,y2), color, 2, r=8)
    name = labels[int(cls)] if 0 <= int(cls) < len(labels) else str(int(cls))
    _label_badge(frame, x1, y1, f"{name} {int(score*100+0.5)}%", bg=color, fg=WHITE)

# ---------- Detection decoders ----------
_GRID_CACHE={}
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
    boxes = arr.reshape(-1,7); dets=[]
    for image_id,label,score,x_min,y_min,x_max,y_max in boxes:
        if score < conf: continue
        dets.append([x_min*img_size,y_min*img_size,x_max*img_size,y_max*img_size,score,label])
    return np.array(dets,np.float32) if dets else np.zeros((0,6),np.float32)

def decode_box5_plus_cls(out0, out1, size, conf=0.35):
    B,N,C = out0.shape; assert B==1 and C==5
    boxes = out0[0].astype(np.float32)
    clsid = out1[0].astype(np.float32) if (out1.ndim==2 and out1.shape[1]==N) else np.zeros((N,),np.float32)
    if np.all((boxes[:,:4]>=0)&(boxes[:,:4]<=1.5)): boxes[:,:4]*=size
    keep = boxes[:,4] >= conf
    if not np.any(keep): return np.zeros((0,6),np.float32)
    b=boxes[keep]; c=clsid[keep]
    return np.concatenate([b[:,:4], b[:,4:5], c[:,None]],1)

def brute_try_any(arr, size, conf=0.1, topk=5):
    A=arr
    if A.ndim==3 and A.shape[0]==1: A=A[0]
    if A.ndim!=2 or A.shape[1]<4 or A.shape[1]>7: return np.zeros((0,6),np.float32)
    X=A.astype(np.float32)
    if np.all((X[:,:4]>=0)&(X[:,:4]<=1.5)): X[:,:4]*=size
    if A.shape[1]==4:
        score=np.ones((X.shape[0],1),np.float32); cls=np.zeros((X.shape[0],1),np.float32)
        X=np.concatenate([X[:,:4],score,cls],1)
    elif A.shape[1]>=5:
        score=X[:,4:5]; cls=X[:,5:6] if A.shape[1]>=6 else np.zeros((X.shape[0],1),np.float32)
        X=np.concatenate([X[:,:4],score,cls],1)
    idx=np.argsort(-X[:,4])[:topk]; Y=X[idx]
    return Y[Y[:,4]>=conf] if np.any(Y[:,4]>=conf) else Y

# ---------- Seg helpers (vivid style) ----------
def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# vivid palette (배경 0, 예: 1=sticker, 2=stain)
SEG_STYLE = {
    1: {"fill":(0,255,255),  "alpha":0.45, "edge":(60,220,60),  "badge":(40,180,40),  "name":"sticker"},
    2: {"fill":(255,120,0),  "alpha":0.45, "edge":(255,210,120),"badge":(200,160,80), "name":"stain"},
}

def draw_seg_on_roi(roi_bgr, mask_uint8, labels):
    """Draw vivid fill + contour + small badge per class (skip background=0)."""
    h,w = mask_uint8.shape
    out = roi_bgr.copy()

    for cid in SEG_STYLE.keys():
        m = (mask_uint8 == cid).astype(np.uint8)
        if m.sum() == 0: continue

        # fill
        color = SEG_STYLE[cid]["fill"]; alpha = SEG_STYLE[cid]["alpha"]
        overlay = out.copy()
        overlay[m>0] = color
        cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)

        # contour
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0: 
            continue
        cv2.drawContours(out, contours, -1, SEG_STYLE[cid]["edge"], 2)

        # badge at largest contour centroid
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas):
            c = contours[int(np.argmax(areas))]
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                # label name
                name = SEG_STYLE[cid]["name"] if cid < len(labels) else labels[cid] if cid < len(labels) else str(cid)
                bg  = SEG_STYLE[cid]["badge"]
                txt = f"{name}"
                font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.5; thick=1; pad=4
                (tw,th), _ = cv2.getTextSize(txt, font, scale, thick)
                bx1,by1 = max(0, cx-tw//2-pad), max(0, cy- th//2 - pad - 12)
                bx2,by2 = min(w-1, bx1 + tw + 2*pad), min(h-1, by1 + th + 2*pad)
                cv2.rectangle(out,(bx1,by1),(bx2,by2),bg,-1)
                cv2.putText(out, txt, (bx1+pad, by2-pad), font, scale, WHITE, 1, cv2.LINE_AA)

    return out

# ---------- Factory Controller ----------
def init_hardware_states(ctrl: FactoryController):
    """
    요구사항:
      - 시작 시 conveyor True
      - 나머지(red/orange/green/led) False
    """
    try:
        # 일부 디바이스에선 속성 미존재 가능 → best-effort
        ctrl.red = False
        ctrl.orange = False
        ctrl.green = False
        ctrl.led = False
    except Exception:
        pass
    ctrl.conveyor = True

# ---------- main ----------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description="Det -> ROI Seg (OpenVINO) + FactoryController")
    # det
    ap.add_argument("--det-xml", required=True, help="Detection IR .xml")
    ap.add_argument("--det-bin", required=True, help="Detection IR .bin")
    ap.add_argument("--det-size", type=int, default=640, help="Detection input size (square)")
    ap.add_argument("--det-conf", type=float, default=0.35, help="Detection score threshold")
    ap.add_argument("--det-labels", type=str, default="cap", help="Comma labels for detection classes")
    # seg (ROI only)
    ap.add_argument("--seg-xml", required=True, help="Segmentation IR .xml")
    ap.add_argument("--seg-bin", required=True, help="Segmentation IR .bin")
    ap.add_argument("--seg-size", type=int, default=512, help="Segmentation input size (square)")
    ap.add_argument("--seg-labels", type=str, default="background,stain,sticker")
    # common
    ap.add_argument("--src", default="/dev/video0", help="Video source: index or path (e.g., /dev/video0)")
    ap.add_argument("--device", default="AUTO", choices=["CPU","GPU","AUTO"], help="OpenVINO device")
    ap.add_argument("--scale", type=float, default=255.0, help="Input scale (divide)")
    ap.add_argument("--print-every", type=int, default=60)
    args = ap.parse_args()

    det_labels = [s.strip() for s in args.det_labels.split(",") if s.strip()]
    seg_labels = [s.strip() for s in args.seg_labels.split(",") if s.strip()]

    # ---- FactoryController 결합 ----
    ctrl = FactoryController(debug=True)
    try:
        init_hardware_states(ctrl)   # 컨베이어 True, 나머지 False
    except Exception as e:
        print(f"[경고] 하드웨어 초기화 문제 (무시하고 계속 진행): {e}")

    # ---- OpenVINO 준비 ----
    ie = Core()
    opts = {
        "PERFORMANCE_HINT":"THROUGHPUT",
        "INFERENCE_PRECISION_HINT":"f16",
        "CACHE_DIR":"./ov_cache",
        "NUM_STREAMS":"AUTO"
    }
    det_net = ie.compile_model(ie.read_model(args.det_xml,args.det_bin), args.device, opts)
    seg_net = ie.compile_model(ie.read_model(args.seg_xml,args.seg_bin), args.device, opts)
    det_outs=[det_net.output(i) for i in range(len(det_net.outputs))]
    seg_out = seg_net.output(0)

    # ---- Video ----
    cap = cv2.VideoCapture(args.src if not str(args.src).isdigit() else int(args.src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.src}")

    t0=time.time(); frames=0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            H,W = frame.shape[:2]

            # ---- DET ----
            d_img, d_r, d_l, d_t = letterbox(frame, args.det_size)
            d_rgb = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
            d_ten = d_rgb.transpose(2,0,1)[None].astype(np.float32)
            if args.scale: d_ten/=args.scale
            d_res = det_net([d_ten]); d_arrs=[d_res[o] for o in det_outs]

            det=None
            # a) (boxes5, cls) 조합
            if len(d_arrs)>=2:
                a0,a1=d_arrs[0],d_arrs[1]
                if a0.ndim==3 and a0.shape[0]==1 and a0.shape[2]==5 and a1.ndim==2 and a1.shape[0]==1:
                    det=decode_box5_plus_cls(a0,a1,args.det_size,args.det_conf)
                elif a1.ndim==3 and a1.shape[0]==1 and a1.shape[2]==5 and a0.ndim==2 and a0.shape[0]==1:
                    det=decode_box5_plus_cls(a1,a0,args.det_size,args.det_conf)
            # b) 직출 Nx6/7
            if det is None or det.size==0:
                for a in d_arrs:
                    if (a.ndim==2 and a.shape[-1]>=6): det=a; break
                    if (a.ndim==3 and a.shape[0]==1 and a.shape[-1]>=6): det=a[0]; break
            # c) SSD (1,1,N,7)
            if det is None or det.size==0:
                for a in d_arrs:
                    if a.ndim==4 and a.shape[-1]==7:
                        det=decode_detection_output(a,args.det_size,args.det_conf); break
            # d) YOLOX raw
            if det is None or det.size==0:
                for a in d_arrs:
                    if a.ndim==3 and a.shape[0]==1 and (a.shape[1]>=6 or a.shape[2]>=6):
                        raw=a[0]
                        pred = raw if (raw.ndim==2 and raw.shape[0]>=6) else (raw.T if (raw.ndim==2 and raw.shape[1]>=6) else None)
                        if pred is None and raw.ndim==3:
                            tmp=raw.reshape(raw.shape[0],-1); pred=tmp if tmp.shape[0]>=6 else tmp.T
                        if pred is not None:
                            det=decode_yolox_raw(pred,args.det_size,args.det_conf); break
            if det is None: det=np.zeros((0,6),np.float32)

            # ---- ROI SEG + Draw ----
            for x1,y1,x2,y2,score,cls in det:
                # de-letterbox
                gx1=int(np.clip((x1-d_l)/d_r,0,W-1)); gy1=int(np.clip((y1-d_t)/d_r,0,H-1))
                gx2=int(np.clip((x2-d_l)/d_r,0,W-1)); gy2=int(np.clip((y2-d_t)/d_r,0,H-1))
                if gx2<=gx1 or gy2<=gy1: continue

                roi = frame[gy1:gy2, gx1:gx2]
                if roi.size==0: continue

                s_img, s_r, s_l, s_t = letterbox(roi, args.seg_size)
                s_rgb = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
                s_ten = s_rgb.transpose(2,0,1)[None].astype(np.float32)
                if args.scale: s_ten/=args.scale
                s_res = seg_net([s_ten])[seg_out]  # [1,C,H,W] or [1,1,H,W]

                if s_res.ndim==4 and s_res.shape[1]>1:
                    prob = softmax(s_res, axis=1)[0]
                    mask  = np.argmax(prob, axis=0).astype(np.uint8)
                else:
                    mask = (s_res[0,0]>0.5).astype(np.uint8)

                # un-letterbox back to roi size
                canvas = np.zeros((args.seg_size, args.seg_size), dtype=np.uint8)
                nh = int(round(roi.shape[0]*s_r)); nw = int(round(roi.shape[1]*s_r))
                top = (args.seg_size-nh)//2; left=(args.seg_size-nw)//2
                mask_rs = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
                canvas[top:top+nh, left:left+nw] = mask_rs
                mask_roi = cv2.resize(canvas,(roi.shape[1],roi.shape[0]),interpolation=cv2.INTER_NEAREST)

                # vivid draw + bbox
                frame[gy1:gy2, gx1:gx2] = draw_seg_on_roi(roi, mask_roi, seg_labels)
                draw_box(frame, gx1, gy1, gx2, gy2, score, cls, det_labels, color=PINK)

                # ---- (옵션) 액추에이터 트리거 예시: 필요 시 규칙 추가 ----
                # if det_labels[int(cls)] == "defect" and score >= 0.6:
                #     ctrl.pulse_actuator_1(120)   # 예: 120ms 펄스 (사용 환경에 맞게 구현)

            # ---- UI ----
            frames+=1; fps=frames/max(1e-6, time.time()-t0)
            put_fps(frame, fps)
            cv2.imshow("Det -> Seg (vivid)  (q to quit)", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    finally:
        cap.release(); cv2.destroyAllWindows()
        try:
            ctrl.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
