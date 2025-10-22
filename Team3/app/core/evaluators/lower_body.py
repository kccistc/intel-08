from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
from .base import ExerciseEvaluator, EvalResult
from .advice import get_advice
from .schema_config import get_dynamic_schema

DEBUG_LOWER = True

def _dbg(*args):
    if DEBUG_LOWER:
        print("[EVAL:lower]", *args)

def _fmt(v):
    if v is None: return "None"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

def _avg_lr(meta: Dict[str, Any], base: str) -> Optional[float]:
    l = meta.get(f"{base}_l_deg"); r = meta.get(f"{base}_r_deg")
    if l is None and r is None: return None
    if l is None: return float(r)
    if r is None: return float(l)
    return (float(l) + float(r)) / 2.0

def _score_by(angle: Optional[float], best: float, maxv: float=None, minv: float=None, **kwargs) -> int:
    if maxv is None and "max" in kwargs:  maxv = kwargs["max"]
    if minv is None and "min" in kwargs:  minv = kwargs["min"]
    if angle is None or math.isnan(angle):
        return 0
    # 원코드 유지: minv < maxv 이면 swap
    if minv < maxv:
        minv, maxv = maxv, minv
    if angle <= maxv:  return 0
    if angle >= minv:  return 100
    ratio = (angle - maxv) / (minv - maxv)
    score = int(round(ratio * 100))
    return max(0, min(100, score))

_LABEL_ALIAS = {
    "squat": "squat", "스쿼트": "squat",
    "leg_raise": "leg_raise", "legraise": "leg_raise", "레그레이즈": "leg_raise",
}
def _normalize_label(lbl: Any) -> Optional[str]:
    if lbl is None: return None
    s = str(lbl).strip().lower().replace("-", "_").replace(" ", "")
    return _LABEL_ALIAS.get(s, s)

# 기본 SCHEMA (폴백)
DEFAULT_SCHEMA: Dict[str, Dict[str, Dict[str, float]]] = {
    "squat": {
        "metrics": {
            "hip":  {"best": 100.0, "maxv": 65.0, "minv": 150.0},
            "knee": {"best": 100.0, "maxv": 60.0, "minv": 150.0},
        },
        "phases": { "down_th": 135.0, "up_th": 165.0, "debounce_n": 3 }
    },
    "leg_raise": {
        "metrics": { "hip": {"best": 140.0, "max": 165.0, "min": 110.0} },
        "phases":  { "down_th": 155.0, "up_th": 170.0, "debounce_n": 3, "cooldown_n": 5 }
    },
}

_cfg = get_dynamic_schema("lower_body", DEFAULT_SCHEMA)

class LowerBodyEvaluator(ExerciseEvaluator):
    """
    ONLY: squat / leg_raise
    - 스쿼트: 카운트=무릎 DOWN→UP, 점수=힙+무릎 최소각 스냅샷
    - 레그 레이즈: 카운트=UP→DOWN, 점수=힙 각도(스냅샷)
    """
    TOL = 3.0

    def __init__(self, label: str = "squat"):
        assert label in ("squat", "leg_raise")
        self.mode = label
        super().__init__()

    def reset(self) -> None:
        self.state = "UP"
        self._deb = 0
        self._passed_up = False
        self._cooldown = 0
        self._last_emit_id = None
        self._min_knee = None
        self._min_hip = None
        _dbg(f"reset() -> state=UP, deb=0, mode={getattr(self,'mode',None)}")

    # ---------- 점수 스냅샷 ----------
    def _score_snapshot(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        if self.mode == "squat":
            return self._score_snapshot_squat(meta)
        else:
            return self._score_snapshot_legraise(meta)

    # ---------- 스쿼트 ----------
    def _update_squat(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        phases = _cfg.get_mode("squat").get("phases", {})
        DOWN_TH = float(phases.get("down_th", 135.0))
        UP_TH   = float(phases.get("up_th",   165.0))
        DEB_N   = int(phases.get("debounce_n", 3))

        knee = _avg_lr(meta, "knee"); hip  = _avg_lr(meta, "hip")

        if knee is None or math.isnan(knee):
            self._deb = 0; return None

        if self.state == "UP":
            if knee < DOWN_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "DOWN"; self._deb = 0
                    self._min_knee = knee; self._min_hip = hip
        else:
            if knee is not None:
                if self._min_knee is None or knee < self._min_knee: self._min_knee = knee
            if hip is not None:
                if self._min_hip  is None or hip  < self._min_hip:  self._min_hip  = hip

            _dbg(f"[SQUAT] DOWN tracking: min_knee={_fmt(self._min_knee)} min_hip={_fmt(self._min_hip)}")

            if knee >= UP_TH:
                self._deb += 1
                if self._deb >= (DEB_N - 1):
                    self.state = "UP"; self._deb = 0
                    meta2 = dict(meta)
                    if self._min_knee is not None:
                        meta2["knee_l_deg"] = meta2["knee_r_deg"] = float(self._min_knee)
                    if self._min_hip is not None:
                        meta2["hip_l_deg"]  = meta2["hip_r_deg"]  = float(self._min_hip)
                    score, used, advice = self._score_snapshot_squat(meta2)
                    self._min_knee = None; self._min_hip = None
                    return EvalResult(rep_inc=1, score=score, advice=advice)
        return None

    def _score_snapshot_squat(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}; scores: List[int] = []
        cfgs = _cfg.get_mode("squat").get("metrics", {})

        hip  = _avg_lr(meta, "hip");  knee = _avg_lr(meta, "knee")
        if hip  is not None and "hip"  in cfgs:  scores.append(_score_by(hip,  **cfgs["hip"]));  used["hip"]  = hip
        if knee is not None and "knee" in cfgs:  scores.append(_score_by(knee, **cfgs["knee"])); used["knee"] = knee

        if not scores: return 50, used, "각도 인식 불가: 프레임/포즈 확인"
        score = int(round(sum(scores) / len(scores)))

        TILT_LIMIT   = 5.0; LUMBAR_LIMIT = 10.0
        ctx = {
            "knee_valgus":      bool(meta.get("knee_valgus", False)),
            "tilt_instability": float(meta.get("torso_jitter", 0.0)) > TILT_LIMIT,
            "back_arch":        float(meta.get("lumbar_ext", 0.0))  > LUMBAR_LIMIT,
        }
        advice = get_advice("squat", score, ctx)
        return score, used, advice

    # ---------- 레그 레이즈 ----------
    def _update_leg_raise(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        phases = _cfg.get_mode("leg_raise").get("phases", {})
        LR_DOWN_TH = float(phases.get("down_th", 155.0))
        LR_UP_TH   = float(phases.get("up_th",   170.0))
        DEB_N      = int(phases.get("debounce_n", 3))
        COOLDOWN_N = int(phases.get("cooldown_n", 5))

        if getattr(self, "_cooldown", 0) > 0:
            self._cooldown -= 1; return None

        hip_r = meta.get("hip_r_deg")
        try: hip_r = float(hip_r) if hip_r is not None else None
        except Exception: hip_r = None

        if hip_r is None or (isinstance(hip_r, float) and math.isnan(hip_r)):
            self._deb = 0; return None

        if self.state == "UP":
            if hip_r >= LR_DOWN_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "DOWN"; self._deb = 0
                    if getattr(self, "_passed_up", False):
                        frame_id = meta.get("frame_id") or meta.get("frame_idx") or meta.get("ts")
                        if frame_id is not None and frame_id == getattr(self, "_last_emit_id", None):
                            return None
                        score, used, advice = self._score_snapshot_legraise(meta)
                        self._last_emit_id = frame_id; self._cooldown = COOLDOWN_N
                        self._passed_up = False
                        return EvalResult(rep_inc=1, score=score, advice=advice)

        else:  # DOWN
            if hip_r <= LR_UP_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "UP"; self._deb = 0; self._passed_up = True
        return None

    def _score_snapshot_legraise(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}
        hip = _avg_lr(meta, "hip"); used["hip"] = hip if hip is not None else float("nan")
        cfg = _cfg.get_mode("leg_raise").get("metrics", {}).get("hip", {"best":140.0, "max":165.0, "min":110.0})
        score = _score_by(hip, **cfg)

        TILT_LIMIT   = 5.0; LUMBAR_LIMIT = 10.0
        ctx = {
            "tilt_instability": float(meta.get("torso_jitter", 0.0)) > TILT_LIMIT,
            "back_arch":        float(meta.get("lumbar_ext", 0.0))  > LUMBAR_LIMIT,
        }
        advice = get_advice("leg_raise", score, ctx)
        return score, used, advice

    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        label = meta.get("label")
        _dbg(
            f"knee(L/R)={_fmt(meta.get('knee_l_deg'))}/{_fmt(meta.get('knee_r_deg'))}, "
            f"hip(L/R)={_fmt(meta.get('hip_l_deg'))}/{_fmt(meta.get('hip_r_deg'))}, "
            f"hipline(L/R)={_fmt(meta.get('hipline_l_deg'))}/{_fmt(meta.get('hipline_r_deg'))}"
        )

        if self.mode == "squat" and label != "squat": return None
        if self.mode == "leg_raise" and label != "leg_raise": return None

        if self.mode == "squat":
            return self._update_squat(meta)
        else:
            return self._update_leg_raise(meta)
