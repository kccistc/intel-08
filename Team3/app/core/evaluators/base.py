from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class EvalResult:
    """
    모든 운동 평가기가 매 프레임마다 반환하는 공통 결과 형식.
    exercise_page는 이 값을 받아 UI(카운트/점수/조언/제목)를 갱신한다.
    """
    rep_inc: int = 0                  # 이번 프레임에서 증가한 횟수 (0 또는 1)
    score: Optional[int] = None       # 점수(0~100 등급) - rep 완료 시 주로 설정
    advice: Optional[str] = None      # 피드백 문구
    title: Optional[str] = None       # 운동 제목 제안(라벨 표시용)


class ExerciseEvaluator(ABC):
    """
    카테고리별(상체/하체/코어) 평가기의 공통 기반 클래스.
    - reset(): 세션/상태 초기화
    - update(meta): 매 프레임 센서/비전 메타를 입력받아 EvalResult 반환
    """
    # 공통 디바운스 기본값(필요 시 하위클래스에서 오버라이드)
    DEBOUNCE_N: int = 3

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """운동 상태(예: UP/DOWN, 프레임 카운터 등)를 초기화한다."""
        ...

    @abstractmethod
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        """
        매 프레임 호출. 카메라/모션 인식 메타(meta)를 바탕으로
        카운트/점수/피드백을 산출하고 EvalResult를 반환한다.
        """
        ...

    # 선형 보간(스코어 계산 등에 사용 가능)
    def _lerp(self, a: float, b: float, t: float) -> float:
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return (1.0 - t) * a + t * b

    # 0~100 점수 범위 클램프
    def _clamp_score(self, s: float) -> int:
        if s < 0:
            return 0
        if s > 100:
            return 100
        return int(round(s))
