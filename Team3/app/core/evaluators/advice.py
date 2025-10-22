from __future__ import annotations
from typing import Dict, List, Optional
import random

# 점수 → 버킷
def score_bucket(score: Optional[int]) -> str:
    s = 0 if score is None else int(score)
    if s <= 50:   return "danger"     # 0~50
    if s <= 70:   return "improve"    # 51~70
    if s <= 89:   return "good"       # 71~89
    return "perfect"                  # 90~100


# 기본 멘트 (운동별/버킷별)
BASE: Dict[str, Dict[str, List[str]]] = {
    "squat": {
        "danger": [
            "엉덩이를 더 뒤로 빼고, 중심을 발뒤꿈치로 옮겨보세요."
        ],
        "improve": [
            "조금만 더 깊이 앉고 코어 긴장 유지."
        ],
        "good": [
            "힙 각도 좋아요."
        ],
        "perfect": [
            "균형과 깊이 모두 훌륭합니다."
        ],
    },

    "leg_raise": {
        "danger": [
            "복부 힘이 풀렸어요. 배꼽을 바닥으로 누르듯 고정!"
        ],
        "improve": [
            "복부로 들어올리세요."
        ],
        "good": [
            "자세 깔끔합니다. 0.5초 정지로 컨트롤 업."
        ],
        "perfect": [
            "코어 컨트롤이 인상적입니다.",
        ],
    },

    "shoulder_press": {
        "danger": [
            "양쪽 팔꿈치를 동일 궤도로 올리세요."
        ],
        "improve": [
            "내릴 때 속도를 낮추고 반동을 줄여보세요.",
        ],
        "good": [
            "어깨 안정성 훌륭합니다."
        ],
        "perfect": [
            "프로 수준, 흔들림 없이 정확한 라인.",
        ],
    },

    "lateral_raise": {
        "danger": [
            "상체 흔들림 없이 올리기.",
        ],
        "improve": [
            "팔꿈치 약간 구부리고 어깨로 들어올리세요.",
        ],
        "good": [
            "대칭이 잘 맞아요.",
        ],
        "perfect": [
            "프로 수준의 레터럴 레이즈입니다."
        ],
    },

    "pushup": {
        "danger": [
            "코어 조이고 몸 일직선으로 해야합니다.",
        ],
        "improve": [
            "팔꿈치 각도 일정하게—천천히 내려가며 제어.",
        ],
        "good": [
            "속도 일정, 범위 정확. 아주 좋습니다."
        ],
        "perfect": [
            "몸 전체가 일직선, 컨트롤 최고.",
        ],
    },

    "burpee": {
        "danger": ["속도를 낮추고 단계별로 정확히!"],
        "improve": ["반동 대신 근육으로 해야합니다."],
        "good": ["리듬 유지하고 코어 조금 더 조이기."],
        "perfect": ["리듬·정확도 모두 훌륭합니다."],
    },

    "jumping_jack": {
        "danger": ["팔/다리 범위가 불안정 합니다."],
        "improve": ["팔·다리   반동 줄이기."],
        "good": ["속도 일정, 범위 유지 하시면 딱 좋아요."],
        "perfect": ["타이밍과 대칭이 훌륭합니다."],
    },

    "generic": {
        "danger": ["속도를 낮추고 정렬을 먼저 교정하세요."],
        "improve": ["범위와 속도를 일정하게 유지해보세요."],
        "good": ["세부 정렬만 다듬으면 완벽해요."],
        "perfect": ["현재 패턴을 유지하세요"],
    },
}

# 상황별 보정 팁(선택적으로 덧붙임)
TIP_MAP: Dict[str, str] = {
    "elbow_not_deep":   "팔꿈치를 더 접어 깊이를 확보하세요.",
    "elbow_flare":      "팔꿈치가 벌어집니다.",
    "knee_more_extend": "무릎을 더 펴 안정적인 라인을 만들어요.",
    "knee_valgus":      "무릎이 안쪽으로 말립니다",
    "hip_back":         "엉덩이를 더 뒤로 빼서 중심을 잡아보세요.",
    "back_arch":        "코어 조이고 중립 유지.",
    "tilt_instability": "상체 흔들림이 커요.",
}

# 즉시 반복 방지 선택기: 직전 선택은 다음 호출에서 제외
# 키: (exercise, bucket)
class _RecentPicker:
    def __init__(self):
        self.last_chosen: Dict[tuple, str] = {}
        self.rng = random.Random()

    def set_seed(self, seed: Optional[int]):
        if seed is not None:
            self.rng.seed(seed)

    def choose(self, exercise: str, bucket: str, candidates: List[str]) -> str:
        if not candidates:
            return ""

        key = (exercise, bucket)
        last = self.last_chosen.get(key)

        # 후보가 1개면 그대로 반환
        if len(candidates) == 1:
            chosen = candidates[0]
            self.last_chosen[key] = chosen
            return chosen

        # 직전 문구 제외한 후보 생성
        pool = [c for c in candidates if c != last] if last in candidates else candidates[:]

        # 만약 모두 제외되어 빈 리스트가 됐다면(동일 문구 1개만 있는 특수 상황)
        if not pool:
            pool = candidates[:]

        chosen = self.rng.choice(pool)
        self.last_chosen[key] = chosen
        return chosen

_picker = _RecentPicker()

def set_random_seed(seed: Optional[int]) -> None:
    """테스트/재현 목적의 시드 고정"""
    _picker.set_seed(seed)

# 내부: 기본 멘트 1개 선택(+즉시 반복 방지)
def _pick_base(exercise: str, bucket: str) -> str:
    ex = exercise if exercise in BASE else "generic"
    arr = BASE.get(ex, {}).get(bucket) or BASE["generic"][bucket]
    return _picker.choose(ex, bucket, arr)

# 공개 API: 최종 코칭 멘트 생성
def get_advice(exercise: str, score: Optional[int], ctx: Optional[Dict[str, bool]] = None) -> str:
    """
    exercise: "squat"|"leg_raise"|"shoulder_press"|"lateral_raise"|"pushup"|...
    score   : 0~100
    ctx     : {"elbow_not_deep": True, "knee_more_extend": True, ...}
    """
    bucket = score_bucket(score)
    base = _pick_base(exercise, bucket)

    if not ctx:
        return base

    tips = [TIP_MAP[k] for k, v in ctx.items() if v and k in TIP_MAP]
    if not tips:
        return base

    # 기본 멘트 + 보정 팁 결합
    return f"{base} " + " ".join(tips)

### 사운드 재생 ### 
def get_advice_with_sfx(
    exercise: str,
    score: Optional[int],
    ctx: Optional[Dict[str, bool]] = None
    ) -> tuple[str, str, str]:
        
    bucket = score_bucket(score)
    text = get_advice(exercise, score, ctx)
    exercise_norm = (exercise or "generic").strip().lower().replace("-", "_")
    sfx_key = f"{exercise_norm}_{bucket}"
    return text, bucket, sfx_key
