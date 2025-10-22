from __future__ import annotations
from typing import Optional
from .base import EvalResult, ExerciseEvaluator

def get_advice_with_sfx(*args, **kwargs):
    from importlib import import_module
    mod = import_module(".advice", __package__)  
    return mod.get_advice_with_sfx(*args, **kwargs)

__all__ = [
    "compute_joint_angles", "update_meta_with_angles",
    "ExerciseEvaluator", "EvalResult",
    "get_evaluator_by_label",
    "get_advice_with_sfx",
]
# ---- 라벨 별칭(한글/대소문자/공백/대시 호환) ----
_ALIAS = {
    # 하체
    "squat": "squat",
    "스쿼트": "squat",
    "legraise": "leg_raise",
    "leg_raise": "leg_raise",
    "leg raise": "leg_raise",
    "leg-raise": "leg_raise",
    "레그레이즈": "leg_raise",
    "레그 레이즈": "leg_raise",

    # 상체
    "pushup": "pushup",
    "푸쉬업": "pushup",
    "shoulderpress": "shoulder_press",
    "shoulder_press": "shoulder_press",
    "숄더프레스": "shoulder_press",
    "side_lateral_raise": "side_lateral_raise",
    "side_lateral": "side_lateral_raise",
    "side_lateral-raise": "side_lateral_raise",
    "사이드레터럴레이즈": "side_lateral_raise",
    "dumbbellrow": "dumbbell_row",
    "dumbbell_row": "dumbbell_row",
    "덤벨로우": "dumbbell_row",

    # 코어
    "burpee": "burpee",
    "버피": "burpee",
}

# ---- 지연 생성용 캐시 ----
_SINGLETONS: dict[str, ExerciseEvaluator] = {}

def _normalize(label: str) -> str:
    key = label.strip().lower().replace("-", "_").replace(" ", "")
    return _ALIAS.get(key, key)

def _create_instance(key: str) -> ExerciseEvaluator:
    # 필요한 순간에만 모듈 임포트 (순환 방지)

    # 하체
    if key in ("squat", "leg_raise"):
        from .lower_body import LowerBodyEvaluator
        return LowerBodyEvaluator(key)

    # 코어  ← 원래 코어가 담당하는 3개: ("burpee", "pushup", "Jumping_jack")
    if key in ("burpee", "pushup", "jumping_jack"):
        try:
            from .core_full import CoreBodyEvaluator 
        except Exception as e:
            raise ImportError(f"CoreFullEvaluator 로드 실패: {e}")
        return CoreBodyEvaluator(key)

    # 상체 
    if key in ("shoulder_press", "side_lateral_raise", "Bentover_Dumbbell", "bentover_dumbbell"):
        from .upper_body import UpperBodyEvaluator
        return UpperBodyEvaluator(key)

    # 알 수 없는 라벨
    raise KeyError(f"Unknown evaluator label: {key}")

def get_evaluator_by_label(label: str) -> Optional[ExerciseEvaluator]:
    """운동 라벨로 평가기 싱글톤을 반환 (지연 import/생성)."""
    if not label:
        return None
    key = _normalize(label)
    inst = _SINGLETONS.get(key)
    if inst is None:
        inst = _create_instance(key)
        _SINGLETONS[key] = inst
    return inst
