from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class Exercise:
    key: str
    title: str
    category: str
    sets_reps: str
    description: str
    goal_muscles: str
    recommend: str
    steps: List[str]
    tips: List[str]
    video: str = ""

def _to_exercise_list(raw_list: List[Dict[str, Any]]) -> List[Exercise]:
    return [Exercise(**d) for d in raw_list]

def list_all() -> List[Exercise]:
    return _to_exercise_list(EXERCISES_DATA)

EXERCISES_DATA: List[Dict[str, Any]] = [
    {
        "key": "squat",
        "title": "스쿼트",
        "category": "하체",
        "sets_reps": "3세트 x 15회",
        "description": "하체 근력 강화를 위한 기본 운동으로, 다리와 엉덩이 근육을 집중적으로 발달시킵니다.",
        "goal_muscles": "대퇴사두근, 둔근",
        "recommend": "3세트 x 15회",
        "steps": [
            "발을 어깨 너비로 벌리고 서세요",
            "무릎이 발끝을 넘지 않도록 주의하며 앉으세요",
            "허벅지가 바닥과 평행이 될 때까지 내려가세요",
            "발뒤꿈치로 밀어내며 일어서세요",
        ],
        "tips": [
            "등을 항상 곧게 유지하세요",
            "무릎은 발끝과 같은 방향으로 향하게 하세요",
            "호흡을 자연스럽게 유지",
        ],
        "video":"assets/videos/jm_guide.mp4",
    },
    {
        "key": "shoulder_press",
        "title": "숄더프레스",
        "category": "어깨",
        "sets_reps": "3세트 x 12회",
        "description": "어깨 전면과 측면 근육을 강화하는 대표적인 프리웨이트 밀기 동작입니다.",
        "goal_muscles": "전면삼각근, 측면삼각근, 상완삼두근",
        "recommend": "3세트 x 12회",
        "steps": [
            "덤벨 또는 바벨을 어깨 높이에서 잡습니다",
            "팔꿈치를 살짝 앞으로 향하게 유지합니다",
            "호흡을 내쉬며 팔을 위로 밀어 올립니다",
            "팔꿈치를 완전히 잠그기 전까지만 올리고 천천히 내립니다",
        ],
        "tips": [
            "허리가 과도하게 꺾이지 않도록 복부에 힘을 줍니다",
            "손목이 꺾이지 않게 수직으로 유지합니다",
            "덤벨을 내릴 때 어깨 근육의 긴장을 유지하세요",
        ],
        "video":"assets/videos/ms_guide.mp4",
    },
    {
        "key": "push_up",
        "title": "푸쉬업",
        "category": "가슴",
        "sets_reps": "3세트 x 15회",
        "description": "상체 전반 근력을 강화하는 대표적인 맨몸 밀기 운동입니다.",
        "goal_muscles": "대흉근, 삼두근, 전면삼각근",
        "recommend": "3세트 x 15회",
        "steps": [
            "어깨 너비보다 약간 넓게 손을 짚고 플랭크 자세를 잡습니다",
            "팔꿈치를 굽혀 몸을 천천히 바닥으로 낮춥니다",
            "가슴이 거의 바닥에 닿기 직전까지 내려갑니다",
            "손바닥으로 밀며 다시 시작 자세로 올라옵니다",
        ],
        "tips": [
            "허리가 꺾이거나 엉덩이가 들리지 않게 몸통을 일직선으로 유지합니다",
            "팔꿈치는 몸통에서 약 45도 각도를 유지합니다",
            "근육의 수축을 느끼며 반동 없이 천천히 수행합니다",
        ],
        "video":"assets/videos/jm_guide.mp4",
    },
    {
        "key": "leg_raise",
        "title": "레그레이즈",
        "category": "복부",
        "sets_reps": "3세트 x 12회",
        "description": "하복부를 집중적으로 자극하는 맨몸 코어 운동입니다.",
        "goal_muscles": "하복부, 고관절 굴곡근",
        "recommend": "3세트 x 12회",
        "steps": [
            "바닥에 누운 상태에서 다리를 곧게 펴고 손은 옆에 둡니다",
            "호흡을 내쉬며 다리를 45도 이상 천천히 들어 올립니다",
            "허리가 뜨지 않도록 복부에 힘을 유지합니다",
            "다리를 바닥에 완전히 닿지 않도록 10cm 위에서 멈춥니다",
        ],
        "tips": [
            "허리가 뜬다면 손을 엉덩이 아래에 받쳐도 좋습니다",
            "반동 없이 천천히 올리고 천천히 내립니다",
            "복부 긴장을 유지하며 호흡을 멈추지 않습니다",
        ],
        "video":"assets/videos/jm_guide.mp4",
    },
    {
        "key": "bent_over_dumbbell_row",
        "title": "벤트오버 덤벨로우",
        "category": "등",
        "sets_reps": "3세트 x 12회",
        "description": "허리 각도를 고정한 상태에서 등 근육을 당기는 프리웨이트 운동입니다.",
        "goal_muscles": "광배근, 능형근, 척추기립근",
        "recommend": "3세트 x 12회",
        "steps": [
            "덤벨을 들고 무릎을 살짝 굽힌 자세에서 상체를 45도 숙입니다",
            "허리를 곧게 펴고 시선은 아래를 향합니다",
            "숨을 들이마시고 덤벨을 복부 쪽으로 끌어당깁니다",
            "광배근의 수축을 느끼며 천천히 덤벨을 내립니다",
        ],
        "tips": [
            "허리가 말리지 않도록 척추 중립을 유지합니다",
            "덤벨은 팔이 아니라 등으로 당긴다는 느낌으로 수행합니다",
            "최대 수축 지점에서 1초 정도 정지하면 자극이 증가합니다",
        ],
        "video":"assets/videos/jm_guide.mp4",
    },
    {
        "key": "side_lateral_raise",
        "title": "사이드 레터럴 레이즈",
        "category": "어깨",
        "sets_reps": "3세트 x 15회",
        "description": "어깨 측면 삼각근을 집중적으로 자극하는 프리웨이트 운동으로 어깨라인을 넓혀주는 데 효과적입니다.",
        "goal_muscles": "측면삼각근",
        "recommend": "3세트 x 15회",
        "steps": [
            "덤벨을 양손에 들고 허벅지 옆에 자연스럽게 둡니다",
            "팔꿈치를 살짝 굽힌 상태를 유지합니다",
            "호흡을 내쉬며 팔을 어깨높이까지 양옆으로 들어 올립니다",
            "천천히 시작 자세로 내려옵니다",
        ],
        "tips": [
            "덤벨을 올릴 때 승모근이 개입되지 않도록 어깨만 사용합니다",
            "반동 없이 천천히 올리고 내립니다",
            "어깨 긴장을 유지하며 팔꿈치 위치를 일정하게 유지합니다",
        ],
        "video":"assets/videos/jm_guide.mp4",
    },
    {
        "key": "burpee",
        "title": "버피",
        "category": "전신",
        "sets_reps": "3세트 x 10회",
        "description": "전신 근육을 동시에 사용하는 고강도 인터벌 운동으로 유산소와 근력 효과를 동시에 얻을 수 있습니다.",
        "goal_muscles": "전신 근육, 심폐지구력",
        "recommend": "3세트 x 10회",
        "steps": [
            "선 자세에서 스쿼트 하듯 앉으며 손을 바닥에 짚습니다",
            "발을 뒤로 튕겨 플랭크 자세로 전환합니다",
            "푸쉬업을 1회 수행합니다 (선택)",
            "다시 발을 당겨 일어서며 점프합니다",
        ],
        "tips": [
            "허리가 꺾이지 않도록 플랭크 자세에서 코어를 조입니다",
            "호흡을 일정하게 유지하며 리듬감 있게 수행합니다",
            "초보자는 푸쉬업 없이 동작을 단순화해도 좋습니다",
        ],
        "video":"assets/videos/jm_guide.mp4",
    },
]
