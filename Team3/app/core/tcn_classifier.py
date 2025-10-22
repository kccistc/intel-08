from __future__ import annotations
import os, json
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Sequence
import numpy as np

NUM_KPTS = 17

def _softmax_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

@dataclass
class TCNConfig:
    classes: Optional[List[str]] = None
    features: str = "xyconf"          
    norm: str = "image"              
    win: int = 60
    input_channels: int = 51          # xyconf=51, xy=34
    kpt_order: Optional[List[int]] = None  # 길이 17, src->dst 인덱스 매핑
    smooth: int = 7                   # EMA window (1이면 비활성)

    @staticmethod
    def from_json(json_path: Optional[str]) -> "TCNConfig":
        if not json_path or not os.path.exists(json_path):
            return TCNConfig()
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}

        hp = cfg.get("hparams", {}) or {}
        kpt_order = cfg.get("kpt_order")
        if not (isinstance(kpt_order, list) and len(kpt_order) == NUM_KPTS):
            kpt_order = None

        features = cfg.get("features", "xyconf")
        default_in_ch = 51 if features == "xyconf" else 34

        return TCNConfig(
            classes=cfg.get("classes"),
            features=features,
            norm=cfg.get("norm", "image"),
            win=int(hp.get("win", cfg.get("win", 60))),
            input_channels=int(cfg.get("input_channels", default_in_ch)),
            kpt_order=kpt_order,
            smooth=int(cfg.get("smooth", 7)),
        )

class TCNOnnxClassifier:
    """
    ONNX TCN 기반 동작 분류기 (CPU 전용).
    - 입력: 최근 T개의 포즈 피처(window)
    - 출력: 클래스 확률, EMA 스무딩 선택적 적용
    """
    def __init__(
        self,
        onnx_path: str,
        json_path: Optional[str] = None,
        session: Any = None,   # onnxruntime.InferenceSession 호환 객체 주입용
    ):
        self.cfg = TCNConfig.from_json(json_path)
        self.classes: Optional[List[str]] = (self.cfg.classes[:] if self.cfg.classes else None)

        self._ema: Optional[np.ndarray] = None
        self._alpha: float = 1.0 if self.cfg.smooth <= 1 else 2.0 / float(self.cfg.smooth + 1)

        self.win: int = int(self.cfg.win)
        self.buf: deque[np.ndarray] = deque(maxlen=self.win)

        self.session = None
        self.input_name: Optional[str] = None
        self.expected_C: Optional[int] = None
        self.expected_T: Optional[int] = None
        self.channels_first: bool = True  # 내부 입력은 [N, C, T]

        if session is not None:
            self.session = session
        else:
            # onnxruntime (CPU) 필요
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
            )

        self.input_name = self.session.get_inputs()[0].name
        self._infer_layout_and_dims()

        model_T = int(self.expected_T or self.win)
        if model_T > self.win:
            self.win = model_T
            self.buf = deque(maxlen=self.win)

        self.ok: bool = True
        self.err: Optional[Exception] = None

    def _infer_layout_and_dims(self) -> None:
        C_json = int(self.cfg.input_channels)
        T_try = int(self.cfg.win)

        def _probe(shape_ntc: bool) -> bool:
            x = np.zeros((1, C_json, T_try), np.float32) if not shape_ntc else np.zeros((1, T_try, C_json), np.float32)
            try:
                _ = self.session.run(None, {self.input_name: x})
                return True
            except Exception:
                return False

        ok_nct = _probe(shape_ntc=False)
        ok_ntc = _probe(shape_ntc=True)

        self.expected_C = C_json
        self.expected_T = T_try if (ok_nct or ok_ntc) else T_try

        out0 = self.session.get_outputs()[0]
        oshape = out0.shape
        if len(oshape) >= 2 and isinstance(oshape[1], int) and oshape[1] > 0:
            n_cls = oshape[1]
            if not self.classes or len(self.classes) != n_cls:
                self.classes = (self.classes or [])[:n_cls]
                while len(self.classes) < n_cls:
                    self.classes.append(f"class_{len(self.classes)}")

    @staticmethod
    def _select_person(people: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not people:
            return None

        def conf_mean(p: Dict[str, Any]) -> float:
            k = p.get("kpt", [])
            if not k:
                return -1.0
            vals = [float(pt[2]) for pt in k if len(pt) >= 3]
            return float(np.mean(vals)) if vals else -1.0

        def area(p: Dict[str, Any]) -> float:
            bbox = p.get("bbox")
            if not bbox or len(bbox) < 4:
                return 0.0
            x1, y1, x2, y2 = bbox
            return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

        return max(people, key=lambda p: (conf_mean(p), area(p)))

    def _reorder_kpts(self, pts: Sequence[Sequence[float]]) -> List[Tuple[float, float, Optional[float]]]:
        if self.cfg.kpt_order is None:
            return [tuple(p) if len(p) >= 3 else (p[0] if len(p) > 0 else 0.0,
                                                  p[1] if len(p) > 1 else 0.0,
                                                  None) for p in pts]

        out = [(0.0, 0.0, None)] * NUM_KPTS
        for i_src, i_dst in enumerate(self.cfg.kpt_order):
            if i_src < len(pts) and 0 <= i_dst < NUM_KPTS:
                p = pts[i_src]
                if len(p) >= 3:
                    out[i_dst] = (float(p[0]), float(p[1]), float(p[2]))
                elif len(p) == 2:
                    out[i_dst] = (float(p[0]), float(p[1]), None)
                elif len(p) == 1:
                    out[i_dst] = (float(p[0]), 0.0, None)
        return out

    def _norm_xy(self, x: float, y: float, w: int, h: int, bbox: Optional[Sequence[float]]) -> Tuple[float, float]:
        if self.cfg.norm == "bbox" and bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            return ((x - x1) / bw, (y - y1) / bh)
        return (x / max(1.0, float(w)), y / max(1.0, float(h)))

    def _feat_xy(self, person: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        w, h = size
        pts = self._reorder_kpts(person.get("kpt", []))
        bbox = person.get("bbox")

        xy = np.zeros((NUM_KPTS * 2,), dtype=np.float32)
        for i in range(NUM_KPTS):
            if i < len(pts):
                x, y, _c = pts[i]
                xn, yn = self._norm_xy(float(x), float(y), w, h, bbox)
                xy[2 * i] = xn
                xy[2 * i + 1] = yn
        return xy

    def _feat_xyconf(self, person: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        w, h = size
        pts = self._reorder_kpts(person.get("kpt", []))
        bbox = person.get("bbox")

        X = np.zeros(NUM_KPTS, np.float32)
        Y = np.zeros(NUM_KPTS, np.float32)
        C = np.ones(NUM_KPTS, np.float32)

        for i in range(NUM_KPTS):
            if i < len(pts):
                x, y, c = pts[i]
                xn, yn = self._norm_xy(float(x), float(y), w, h, bbox)
                X[i] = xn
                Y[i] = yn
                if c is not None:
                    C[i] = float(c)

        xy_interleaved = np.empty(NUM_KPTS * 2, np.float32)
        xy_interleaved[0::2] = X
        xy_interleaved[1::2] = Y
        return np.concatenate([xy_interleaved, C], axis=0)

    def _make_feat_vec(self, person: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        if self.cfg.features == "xy":
            vec = self._feat_xy(person, size)
        else:
            vec = self._feat_xyconf(person, size)

        C = int(self.expected_C or self.cfg.input_channels or len(vec))
        if vec.shape[0] != C:
            out = np.zeros((C,), dtype=np.float32)
            n = min(C, vec.shape[0])
            out[:n] = vec[:n]
            vec = out
        return vec.astype(np.float32)

    def reset(self) -> None:
        self.buf.clear()
        self._ema = None

    def update(self, people: List[Dict[str, Any]], size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        if self.ok is False or self.session is None:
            return None

        person = self._select_person(people)
        if person is None:
            self.reset()
            return None

        feat = self._make_feat_vec(person, size)
        self.buf.append(feat)

        T_need = int(self.expected_T or self.win)
        if len(self.buf) < T_need:
            return None

        window = np.stack(list(self.buf)[-T_need:], axis=0)  # (T,C)
        x = np.transpose(window[None, ...], (0, 2, 1)).astype(np.float32)  # (1,C,T)

        try:
            out = self.session.run(None, {self.input_name: x})
        except Exception:
            return None

        logits = np.asarray(out[0], dtype=np.float32)

        if logits.ndim == 2:
            probs = _softmax_axis(logits, axis=1)
        elif logits.ndim == 3:
            if self.classes and logits.shape[1] == len(self.classes):
                probs_t = _softmax_axis(logits, axis=1)   # (N,C,T)
                probs = probs_t.mean(axis=2)              # (N,C)
            elif self.classes and logits.shape[2] == len(self.classes):
                probs_t = _softmax_axis(logits, axis=2)   # (N,T,C)
                probs = probs_t.mean(axis=1)              # (N,C)
            else:
                probs = _softmax_axis(logits, axis=-1).mean(axis=-2)
        else:
            probs = _softmax_axis(logits, axis=-1)

        n_cls = probs.shape[-1]
        if not self.classes or len(self.classes) != n_cls:
            self.classes = (self.classes or [])[:n_cls]
            while len(self.classes) < n_cls:
                self.classes.append(f"class_{len(self.classes)}")

        probs_1d = probs[0]

        if self.cfg.smooth > 1:
            if self._ema is None:
                self._ema = probs_1d
            else:
                self._ema = (1.0 - self._alpha) * self._ema + self._alpha * probs_1d
            probs_1d = self._ema

        idx = int(np.argmax(probs_1d))
        score = float(probs_1d[idx])
        return {"label": self.classes[idx], "score": score, "probs": probs_1d}
