from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from .base import ExerciseEvaluator, EvalResult
from .advice import get_advice
from pathlib import Path
from .schema_config import get_dynamic_schema

DEBUG_UPPER = True

# 라벨 정규화
_LABEL_ALIAS = {
    "shoulder_press": "shoulder_press", "shoulderpress": "shoulder_press", "숄더프레스": "shoulder_press",
    "side_lateral_raise": "side_lateral_raise", "sidelateralraise": "side_lateral_raise",
    "side_lateral": "side_lateral_raise", "side_lateral-raise": "side_lateral_raise",
    "dumbbell_row": "dumbbell_row", "dumbbellrow": "dumbbell_row", "덤벨로우": "dumbbell_row",
    "side_lateral_raise(r)": "side_lateral_raise", "side_lateral_raise(l)": "side_lateral_raise",
    "side_lateral_raise_r": "side_lateral_raise", "side_lateral_raise_l": "side_lateral_raise",
    "dumbbell_row(r)": "dumbbell_row", "dumbbell_row(l)": "dumbbell_row",
}
def _normalize_label(lbl: Optional[str]) -> Optional[str]:
    if not lbl: return None
    s = str(lbl).strip().lower().replace("-", "_").replace(" ", "")
    return _LABEL_ALIAS.get(s, s)

# 표시/스네이크 동기화
SNAKE_MAP = {
    "Knee(L)": "knee_l_deg",       "Knee(R)": "knee_r_deg",
    "Hip(L)": "hip_l_deg",         "Hip(R)": "hip_r_deg",
    "Shoulder(L)": "shoulder_l_deg","Shoulder(R)": "shoulder_r_deg",
    "Elbow(L)": "elbow_l_deg",     "Elbow(R)": "elbow_r_deg",
    "HipLine(L)": "hipline_l_deg", "HipLine(R)": "hipline_r_deg",
    "Knee": "knee_avg_deg",        "Hip": "hip_avg_deg",
    "Shoulder": "shoulder_avg_deg","Elbow": "elbow_avg_deg",
    "HipLine": "hipline_avg_deg",
}
def _put_angle(meta: Dict[str, Any], display_key: str, value: Optional[float]) -> None:
    if value is None:
        return
    try:
        v = float(value)
    except Exception:
        return
    meta[display_key] = v
    sk = SNAKE_MAP.get(display_key)
    if sk:
        meta[sk] = v

def _get_first(meta: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None

def _avg_lr(meta: Dict[str, Any], base: str) -> Optional[float]:
    cand_l = [f"{base}_l", f"{base}_L", f"{base}(L)", base.capitalize()+"(L)", f"{base}_l_deg"]
    cand_r = [f"{base}_r", f"{base}_R", f"{base}(R)", base.capitalize()+"(R)", f"{base}_r_deg"]
    l = _get_first(meta, cand_l); r = _get_first(meta, cand_r)
    if l is None and r is None: return None
    if l is None: return r
    if r is None: return l
    return (l + r) / 2.0

def _right_only(meta: Dict[str, Any], base: str) -> Optional[float]:
    return _get_first(meta, [f"{base}_r", f"{base}_R", f"{base}(R)", base.capitalize()+"(R)", f"{base}_r_deg"])

# ---------- 기본 SCHEMA (폴백) ----------
DEFAULT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "shoulder_press": {
        "metrics": { "elbow": { "shape": "triangular", "min": 80.0, "best": 100.0, "max": 160.0 } },
        "phases":  { "up_th": 125.0, "down_th": 110.0, "debounce_n": 2, "cooldown_n": 6 },
    },
    "side_lateral_raise": {
        "metrics": { "shoulder": { "shape": "triangular", "min": 40.0, "best": 60.0, "max": 80.0 } },
        "phases":  { "up_th": 80.0, "down_th": 60.0, "debounce_n": 2, "cooldown_n": 4 },
    },
    "dumbbell_row": {
        "metrics": {
            "shoulder": { "shape": "triangular", "min": 18.0, "best": 30.0, "max": 45.0 },
            "elbow":    { "shape": "plateau",    "min": 145.0, "max": 165.0 },
        },
        "phases":  { "down_th": 145.0, "up_th": 165.0, "debounce_n": 2, "cooldown_n": 0 },
    },
}

# 동적 스키마 핸들러 (upper_body 섹션)
_cfg = get_dynamic_schema("upper_body", DEFAULT_SCHEMA)

def _get_schema(mode: str) -> Dict[str, Any]:
    return _cfg.get_mode(mode)

# 범용 스코어러
def _score_metric(angle: Optional[float], rule: Dict[str, float]) -> int:
    if angle is None:
        return 0
    shape = rule.get("shape", "triangular")

    if shape == "triangular":
        best = float(rule.get("best", 0.0))
        mn   = float(rule.get("min", best - 20.0))
        mx   = float(rule.get("max", best + 20.0))
        if angle <= mn or angle >= mx:
            return 0
        span = (mx - mn) / 2.0
        dist = abs(angle - best)
        score = 100.0 * (1.0 - dist / span)
        return max(0, min(100, int(round(score))))

    elif shape == "plateau":
        mn = float(rule["min"]); mx = float(rule["max"])
        if angle < mn: return 0
        if mn <= angle <= mx:
            return int(80 + 20 * (angle - mn) / (mx - mn))
        return 100 if angle > mx else 0

    return 0

# -------------------- UpperBodyEvaluator --------------------
class UpperBodyEvaluator(ExerciseEvaluator):
    DEBOUNCE_N = 2
    TOL = 3.0

    def __init__(self, label: str):
        norm = _normalize_label(label)
        self.mode = {"Side_lateral_raise":"side_lateral_raise",
                     "Dumbbell_Row":"dumbbell_row"}.get(label, norm)
        super().__init__()

    def reset(self):
        self.state = "DOWN" if getattr(self, "mode", None) == "shoulder_press" else "UP"
        self._deb = 0; self._cooldown = 0
        self._ema_sh = None; self._ema_hl = None
        self._armed = False; self._top_el = None

    # ---------- 스냅샷 스코어/조언 ----------
    def _snapshot_score(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}; scores: List[int] = []
        cfg_all = _get_schema(self.mode)
        metric_cfgs: Dict[str, Dict[str, float]] = cfg_all.get("metrics", {})

        TILT_LIMIT   = 5.0
        LUMBAR_LIMIT = 10.0
        torso_jitter = float(meta.get("torso_jitter", 0.0))
        lumbar_ext   = float(meta.get("lumbar_ext", 0.0))
        elbow_flare  = bool(meta.get("elbow_flare", False))

        if self.mode == "shoulder_press":
            e = _avg_lr(meta, "elbow"); _put_angle(meta, "Elbow", e)
            if e is not None and "elbow" in metric_cfgs:
                scores.append(_score_metric(e, metric_cfgs["elbow"])); used["elbow"] = e
            ctx = {"elbow_flare": elbow_flare,
                   "tilt_instability": torso_jitter > TILT_LIMIT,
                   "back_arch": lumbar_ext > LUMBAR_LIMIT,
                   "elbow_not_deep": (e is not None and e > 120.0)}

        elif self.mode == "side_lateral_raise":
            s = _avg_lr(meta, "shoulder"); _put_angle(meta, "Shoulder", s)
            if s is not None and "shoulder" in metric_cfgs:
                scores.append(_score_metric(s, metric_cfgs["shoulder"])); used["shoulder"] = s
            ctx = {"elbow_flare": elbow_flare,
                   "tilt_instability": torso_jitter > TILT_LIMIT,
                   "back_arch": lumbar_ext > LUMBAR_LIMIT}

        elif self.mode == "dumbbell_row":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            _put_angle(meta, "Shoulder", s); _put_angle(meta, "Elbow", e)
            if s is not None and "shoulder" in metric_cfgs:
                scores.append(_score_metric(s, metric_cfgs["shoulder"])); used["shoulder"] = s
            if e is not None and "elbow" in metric_cfgs:
                scores.append(_score_metric(e, metric_cfgs["elbow"])); used["elbow"] = e
            ctx = {"tilt_instability": torso_jitter > TILT_LIMIT,
                   "back_arch": lumbar_ext > LUMBAR_LIMIT}
        else:
            return 50, {}, "지원되지 않는 운동입니다."

        score = int(round(sum(scores) / max(1, len(scores))))
        advice = get_advice(_advice_key(self.mode), score, ctx)
        return score, used, advice

    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        m = self.mode
        if m == "shoulder_press":    return self._update_shoulder_press(meta)
        if m == "side_lateral_raise":return self._update_side_lateral(meta)
        if m == "dumbbell_row":      return self._update_dumbbell_row(meta)
        return None

    def update_and_maybe_score(self, meta: Dict[str, Any], label: Optional[str] = None) -> Optional[EvalResult]:
        if label: self.mode = _normalize_label(label)
        return self.update(meta)

    # --- Shoulder Press ---
    def _update_shoulder_press(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        phases = _get_schema("shoulder_press").get("phases", {})
        EL_UP_TH   = float(phases.get("up_th", 125.0))
        EL_DOWN_TH = float(phases.get("down_th", 110.0))
        DEB_N      = int(phases.get("debounce_n", 2))
        COOLDOWN_N = int(phases.get("cooldown_n", 6))

        el = _avg_lr(meta, "elbow")
        if el is None: self._deb = 0; return None
        _put_angle(meta, "Elbow", el)

        if not hasattr(self, "_cooldown"): self._cooldown = 0
        if self._cooldown > 0: self._cooldown -= 1

        def bump(cond: bool):
            self._deb = min(self._deb + 1, DEB_N) if cond else max(self._deb - 1, 0)

        if self.state == "DOWN":
            bump(el >= EL_UP_TH)
            if self._deb >= DEB_N and self._cooldown == 0:
                self.state = "UP"; self._deb = 0; self._top_el = el
        else:
            if el is not None:
                self._top_el = max(self._top_el or el, el)
            bump(el <= EL_DOWN_TH)
            if self._deb >= DEB_N:
                self.state = "DOWN"; self._deb = 0; self._cooldown = COOLDOWN_N
                meta2 = dict(meta)
                if getattr(self, "_top_el", None) is not None:
                    meta2["elbow_l_deg"] = meta2["elbow_r_deg"] = float(self._top_el)
                score, used, advice = self._snapshot_score(meta2)
                self._top_el = None
                return EvalResult(rep_inc=1, score=score, advice=advice)
        return None

    # --- Side Lateral Raise ---
    def _update_side_lateral(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        phases = _get_schema("side_lateral_raise").get("phases", {})
        SH_UP_TH   = float(phases.get("up_th", 80.0))
        SH_DOWN_TH = float(phases.get("down_th", 60.0))
        DEB_N      = int(phases.get("debounce_n", 2))
        COOLDOWN_N = int(phases.get("cooldown_n", 4))

        s = _avg_lr(meta, "shoulder")
        if s is None: self._deb = 0; return None
        _put_angle(meta, "Shoulder", s)

        if not hasattr(self, "_cooldown"): self._cooldown = 0
        if self._cooldown > 0: self._cooldown -= 1

        if self.state == "DOWN":
            if s >= SH_UP_TH:
                self._deb += 1
                if self._deb >= DEB_N and self._cooldown == 0:
                    self.state = "UP"; self._deb = 0
            else:
                self._deb = 0
        else:
            if s <= SH_DOWN_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "DOWN"; self._deb = 0; self._cooldown = COOLDOWN_N
                    meta2 = dict(meta)
                    meta2["shoulder_l_deg"] = meta2["shoulder_r_deg"] = float(s)
                    score, used, advice = self._snapshot_score(meta2)
                    return EvalResult(rep_inc=1, score=score, advice=advice)
            else:
                self._deb = 0
        return None

    # --- Dumbbell Row ---
    def _update_dumbbell_row(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        phases = _get_schema("dumbbell_row").get("phases", {})
        EL_DOWN_TH = float(phases.get("down_th", 145.0))
        EL_UP_TH   = float(phases.get("up_th", 165.0))
        DEB_N      = int(phases.get("debounce_n", 2))

        elbow = _avg_lr(meta, "elbow")
        if elbow is None: self._deb = 0; return None

        if self.state == "UP":
            if elbow <= EL_DOWN_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "DOWN"; self._deb = 0
            else:
                self._deb = 0
        else:
            if elbow >= EL_UP_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "UP"; self._deb = 0
                    score, used, advice = self._snapshot_score(meta)
                    return EvalResult(rep_inc=1, score=score, advice=advice)
            else:
                self._deb = 0
        return None

def _advice_key(mode: str) -> str:
    return {
        "shoulder_press": "shoulder_press",
        "side_lateral_raise": "side_lateral_raise",
        "dumbbell_row": "dumbbell_row",
    }.get(mode, mode)
