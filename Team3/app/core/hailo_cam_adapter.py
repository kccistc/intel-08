from __future__ import annotations
import math, threading
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import settings as S
from .hailo_pose_stream import start_stream, read_latest, stop_stream

# COCO-17 인덱스
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

def _pt(pts, idx, thr):
    if idx >= len(pts): return None
    x, y = int(pts[idx][0]), int(pts[idx][1])
    c = float(pts[idx][2]) if len(pts[idx]) >= 3 else 1.0
    return (x, y) if c >= thr else None

def _angle(a, b, c) -> Optional[float]:
    if a is None or b is None or c is None: return None
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax-bx, ay-by], dtype=np.float32)
    v2 = np.array([cx-bx, cy-by], dtype=np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    cosv = float(np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))

class HailoCamAdapter:
    def __init__(self, conf_thr: float = 0.65, stride: int = 1,
                 onnx_path: str | None = None, json_path: str | None = None):
        self.conf_thr = conf_thr if conf_thr <= 1.5 else conf_thr/100.0
        self.stride = int(max(1, stride))
        self._lock = threading.Lock()
        self._frame_rgb: Optional[np.ndarray] = None
        self._people: List[Dict[str, Any]] = []
        self._cls: Optional[Dict[str, Any]] = None
        self._size: Tuple[int,int] = (S.SRC_WIDTH, S.SRC_HEIGHT)
        self._running = False

        self._tcn_onnx = onnx_path or getattr(S, "TCN_ONNX", None)
        self._tcn_json = json_path or getattr(S, "TCN_JSON", None)

    def start(self):
        if self._running: return
        kwargs = dict(conf_thr=self.conf_thr, stride=self.stride)
        if self._tcn_onnx and self._tcn_json:
            kwargs.update(onnx_path=self._tcn_onnx, json_path=self._tcn_json)
        start_stream(**kwargs); self._running = True

    def stop(self):
        if not self._running: return
        stop_stream(); self._running = False

    def _pull_once(self) -> bool:
        fr, people, cls, size = read_latest(timeout=0.01)
        if fr is None: return False
        with self._lock:
            self._frame_rgb = fr
            self._people = people
            self._cls = cls
            self._size = size
        return True

    def frame(self) -> Optional[np.ndarray]:
        self._pull_once()
        with self._lock:
            return None if self._frame_rgb is None else self._frame_rgb.copy()

    def people(self) -> List[Dict[str, Any]]:
        self._pull_once()
        with self._lock:
            return list(self._people)

    def meta(self) -> Dict[str, Any]:
        self._pull_once()
        with self._lock:
            ok = bool(self._people)
            w, h = self._size
            label = self._cls.get("label") if isinstance(self._cls, dict) else None
            score = float(self._cls.get("score")) if isinstance(self._cls, dict) and "score" in self._cls else None

            knees = (None, None)
            shoulders = (None, None)
            elbows = (None, None)
            hips = (None, None)
            hiplines = (None, None)

            if self._people:
                p = self._people[0]
                pts = p.get("kpt", [])

                l_knee = _angle(_pt(pts, L_HIP, self.conf_thr), _pt(pts, L_KNE, self.conf_thr), _pt(pts, L_ANK, self.conf_thr))
                r_knee = _angle(_pt(pts, R_HIP, self.conf_thr), _pt(pts, R_KNE, self.conf_thr), _pt(pts, R_ANK, self.conf_thr))
                knees = (l_knee, r_knee)

                l_sho = _angle(_pt(pts, L_HIP, self.conf_thr), _pt(pts, L_SHO, self.conf_thr), _pt(pts, L_ELB, self.conf_thr))
                r_sho = _angle(_pt(pts, R_HIP, self.conf_thr), _pt(pts, R_SHO, self.conf_thr), _pt(pts, R_ELB, self.conf_thr))
                shoulders = (l_sho, r_sho)

                l_elb = _angle(_pt(pts, L_SHO, self.conf_thr), _pt(pts, L_ELB, self.conf_thr), _pt(pts, L_WRI, self.conf_thr))
                r_elb = _angle(_pt(pts, R_SHO, self.conf_thr), _pt(pts, R_ELB, self.conf_thr), _pt(pts, R_WRI, self.conf_thr))
                elbows = (l_elb, r_elb)

                l_hip = _angle(_pt(pts, L_SHO, self.conf_thr), _pt(pts, L_HIP, self.conf_thr), _pt(pts, L_KNE, self.conf_thr))
                r_hip = _angle(_pt(pts, R_SHO, self.conf_thr), _pt(pts, R_HIP, self.conf_thr), _pt(pts, R_KNE, self.conf_thr))
                hips = (l_hip, r_hip)

                l_hl = _angle(_pt(pts, L_SHO, self.conf_thr), _pt(pts, L_HIP, self.conf_thr), _pt(pts, R_HIP, self.conf_thr))
                r_hl = _angle(_pt(pts, R_SHO, self.conf_thr), _pt(pts, R_HIP, self.conf_thr), _pt(pts, L_HIP, self.conf_thr))
                hiplines = (l_hl, r_hl)

            return {
                "ok": ok,
                "src_w": w, "src_h": h,
                "label": label,
                "score": score,
                "knee_l_deg": knees[0], "knee_r_deg": knees[1],
                "shoulder_l_deg": shoulders[0], "shoulder_r_deg": shoulders[1],
                "elbow_l_deg": elbows[0], "elbow_r_deg": elbows[1],
                "hip_l_deg": hips[0], "hip_r_deg": hips[1],
                "hipline_l_deg": hiplines[0], "hipline_r_deg": hiplines[1],
            }
