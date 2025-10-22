from __future__ import annotations
from typing import Dict, Any, Optional
import math
from .base import ExerciseEvaluator, EvalResult
try:
    from .advice import get_advice
except Exception:
    from advice import get_advice
from .schema_config import get_dynamic_schema

DEBUG_CORE = False
def _dbg(*args):
    if DEBUG_CORE:
        print("[EVAL:core]", *args)

def _avg_lr(meta: Dict[str, Any], base: str) -> Optional[float]:
    l = meta.get(f"{base}_l_deg"); r = meta.get(f"{base}_r_deg")
    if l is None and r is None: return None
    if l is None: return float(r)
    if r is None: return float(l)
    return (float(l) + float(r)) / 2.0

# -------- 폴백 SCHEMA (JSON 부재/에러 시) --------
DEFAULT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "burpee": {
        "phases": { "down_enter": 110.0, "up_enter": 110.0 },
        "score":  { "fixed": 100 }
    },
    "pushup": {
        "phases": { "down_enter": 120.0, "score_trigger": 135.0, "debounce_n": 3 },
        "score":  {
            "elbow_min": 90.3, "elbow_max": 133.5,   # 작을수록 좋음
            "knee_min":  90.0, "knee_max": 157.0,    # 클수록 좋음
            "w_elbow": 0.7, "w_knee": 0.3
        }
    },
    "jumping_jack": {
        "phases": { "open_enter": 90.0, "close_enter": 90.0, "count_every": 2 },
        "score":  { "fixed": 100 }
    },
}

# JSON 핫리로드 핸들러
_cfg = get_dynamic_schema("core", DEFAULT_SCHEMA)

# -------- Core 종목 평가기 --------
class CoreBodyEvaluator(ExerciseEvaluator):
    """
    ONLY: burpee / pushup / jumping_jack
    - burpee: 어깨각 임계 기반 카운트
    - pushup: 팔꿈치/무릎 최소각 스냅샷 기반 점수
    - jumping_jack: OPEN↔CLOSE 에지 통과 기반 카운트
    """

    def __init__(self, label: str = "pushup"):
        assert label in ("burpee", "pushup", "jumping_jack")
        self.mode = label
        super().__init__()
        self.complit = False
        self.do = False

        # pushup 상태
        self._pu_state = "UP"
        self._pu_min_elbow = None
        self._pu_min_knee  = None

    def reset(self) -> None:
        self.prev_label = getattr(self, "mode", None)
        self.state = "UP"
        self._deb = 0

        self._pu_state = "UP"
        self._pu_min_elbow = None
        self._pu_min_knee  = None

        _dbg(f"reset() mode={getattr(self,'mode', None)}")

    # ---------- 공개 API ----------
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        m = self.mode  # "burpee" | "pushup" | "jumping_jack"

        if m == "burpee":
            res = self._update_burpee(meta)
            return res if res is not None else EvalResult()

        if m == "pushup":
            res = self._update_pushup(meta)
            return res if res is not None else EvalResult()

        if m == "jumping_jack":
            res = self._update_Jumping_jack(meta)
            return res if res is not None else EvalResult()

        return EvalResult()

    # ---------- burpee ----------
    def _update_burpee(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        s = meta.get("shoulder_r_deg")
        try:
            s = float(s)
        except Exception:
            self.prev_label = "burpee"; return None
        if not math.isfinite(s):
            self.prev_label = "burpee"; return None

        phases = _cfg.get_mode("burpee").get("phases", {})
        DOWN_ENTER = float(phases.get("down_enter", 110.0))
        UP_ENTER   = float(phases.get("up_enter",   110.0))

        state  = getattr(self, "_bp_state", "EXPECT_DOWN")
        prev_s = getattr(self, "_bp_prev_s", s)

        if state == "EXPECT_DOWN":
            if (prev_s > DOWN_ENTER) and (s <= DOWN_ENTER):
                state = "EXPECT_UP"
            self._bp_state = state
            self._bp_prev_s = s
            self.prev_label = "burpee"
            return None

        if (prev_s < UP_ENTER) and (s >= UP_ENTER):
            self._bp_state = "EXPECT_DOWN"
            self._bp_prev_s = s
            self.prev_label = "burpee"
            fixed = int(_cfg.get_mode("burpee").get("score", {}).get("fixed", 100))
            advice_text = get_advice("jumping_jack", fixed, ctx=None)
            return EvalResult(rep_inc=1, score=fixed, advice=advice_text, title="버피")

        self._bp_state = state
        self._bp_prev_s = s
        self.prev_label = "burpee"
        return None

    # ---------- pushup ----------
    def _update_pushup(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        e = meta.get("elbow_r_deg")
        k = meta.get("knee_r_deg")

        try:
            e = float(e)
        except Exception:
            self.prev_label = "pushup"; return None
        if not math.isfinite(e):
            self.prev_label = "pushup"; return None
        if k is not None:
            try: k = float(k)
            except Exception: k = None

        phases = _cfg.get_mode("pushup").get("phases", {})
        DOWN_ENTER    = float(phases.get("down_enter", 120.0))
        SCORE_TRIGGER = float(phases.get("score_trigger", 135.0))
        DEB_N         = int(phases.get("debounce_n", 3))

        if self._pu_state == "UP":
            if e <= DOWN_ENTER:
                self._deb = getattr(self, "_deb", 0) + 1
                if self._deb >= DEB_N:
                    self._pu_state = "DOWN"
                    self._pu_min_elbow = e
                    self._pu_min_knee  = k
                    self._deb = 0
            else:
                self._deb = 0
            self.prev_label = "pushup"
            return None

        # state == "DOWN": 한 rep 동안 최소값 갱신
        if e < self._pu_min_elbow:
            self._pu_min_elbow = e
        if k is not None:
            if (self._pu_min_knee is None) or (k < self._pu_min_knee):
                self._pu_min_knee = k

        if e >= SCORE_TRIGGER:
            self._deb = getattr(self, "_deb", 0) + 1
            if self._deb < DEB_N:
                self.prev_label = "pushup"; return None
            self._deb = 0

            em = self._pu_min_elbow
            km = self._pu_min_knee

            score_cfg = _cfg.get_mode("pushup").get("score", {})
            e_min = float(score_cfg.get("elbow_min", 90.3))
            e_max = float(score_cfg.get("elbow_max", 133.5))
            k_min = float(score_cfg.get("knee_min",  90.0))
            k_max = float(score_cfg.get("knee_max", 157.0))
            w_e   = float(score_cfg.get("w_elbow", 0.7))
            w_k   = float(score_cfg.get("w_knee", 0.3))

            # 팔꿈치 점수 (작을수록 좋음)
            if em < e_min:  em = e_min
            if em > e_max:  em = e_max
            elbow_s = (e_max - em) / (e_max - e_min) * 100.0

            # 무릎 점수 (클수록 좋음)
            if km is not None:
                if km < k_min: km = k_min
                if km > k_max: km = k_max
                knee_s = (km - k_min) / (k_max - k_min) * 100.0
            else:
                knee_s = None

            if knee_s is None:
                final_score = int(round(elbow_s))
            else:
                final_score = int(round(w_e * elbow_s + w_k * knee_s))
            final_score = max(0, min(100, final_score))

            elbow_flare   = bool(meta.get("elbow_flare", False))
            torso_jitter  = float(meta.get("torso_jitter", 0.0))
            lumbar_ext    = float(meta.get("lumbar_ext", 0.0))
            TILT_LIMIT   = 5.0; LUMBAR_LIMIT = 10.0
            ctx = {
                "elbow_not_deep":   (elbow_s < 60.0),
                "elbow_flare":      elbow_flare,
                "knee_more_extend": (knee_s is not None and knee_s < 80),
                "tilt_instability": (torso_jitter > TILT_LIMIT),
                "back_arch":        (lumbar_ext  > LUMBAR_LIMIT),
            }
            advice_text = get_advice("pushup", final_score, ctx)

            self._pu_state = "UP"
            self._pu_min_elbow = None
            self._pu_min_knee  = None

            return EvalResult(rep_inc=1, score=final_score, advice=advice_text)

        self.prev_label = "pushup"
        return None

    # ---------- jumping jack ----------
    def _update_Jumping_jack(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        s = meta.get("shoulder_avg_deg")
        if s is None:
            s = _avg_lr(meta, "shoulder")
        try:
            s = float(s)
        except Exception:
            self.prev_label = "jumping_jack"; return None
        if not math.isfinite(s):
            self.prev_label = "jumping_jack"; return None

        phases = _cfg.get_mode("jumping_jack").get("phases", {})
        OPEN_ENTER  = float(phases.get("open_enter",  90.0))
        CLOSE_ENTER = float(phases.get("close_enter", 90.0))
        COUNT_EVERY = int(phases.get("count_every", 2))

        state  = getattr(self, "_jj_state", "CLOSE")
        cycles = getattr(self, "_jj_cycles", 0)
        prev_s = getattr(self, "_jj_prev_s", s)

        if state == "CLOSE":
            if (prev_s < OPEN_ENTER) and (s >= OPEN_ENTER):
                state = "OPEN"
            self._jj_state = state
            self._jj_cycles = cycles
            self._jj_prev_s = s
            self.prev_label = "jumping_jack"
            return None

        if (prev_s > CLOSE_ENTER) and (s <= CLOSE_ENTER):
            state = "CLOSE"
            cycles += 1
            self._jj_state = state
            self._jj_cycles = cycles
            self._jj_prev_s = s
            self.prev_label = "jumping_jack"

            if (cycles % COUNT_EVERY) == 0:
                fixed = int(_cfg.get_mode("jumping_jack").get("score", {}).get("fixed", 100))
                return EvalResult(rep_inc=1, score=fixed, advice="굿.", title="점핑 잭")
            return None

        self._jj_state = state
        self._jj_cycles = cycles
        self._jj_prev_s = s
        self.prev_label = "jumping_jack"
        return None
