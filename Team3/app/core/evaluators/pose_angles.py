from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import math

__all__ = [
    "compute_joint_angles",
    "update_meta_with_angles",
]

# 내부 유틸
def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """
    각도 ABC(중심 B) in degrees, 범위 0~180.
    실패/불가 시 None 반환 (NaN 사용하지 않음).
    """
    try:
        ba = a - b
        bc = c - b
        nba = float(np.linalg.norm(ba))
        nbc = float(np.linalg.norm(bc))
        if nba < 1e-6 or nbc < 1e-6:
            return None
        cosv = float(np.dot(ba, bc) / (nba * nbc))
        # 수치오차 방지
        cosv = max(-1.0, min(1.0, cosv))
        ang = math.degrees(math.acos(cosv))
        return ang if math.isfinite(ang) else None
    except Exception:
        return None

def _ok(kxy: np.ndarray, kcf: np.ndarray, idxs, thr: float) -> bool:
    """
    좌표는 음수 가능(크롭/오프셋 고려). NaN/Inf 배제 + conf 체크.
    """
    try:
        for i in idxs:
            if not (i < kxy.shape[0] and i < kcf.shape[0]):
                return False
            if not (_is_finite_number(kxy[i, 0]) and _is_finite_number(kxy[i, 1])):
                return False
            if float(kcf[i]) < thr:
                return False
        return True
    except Exception:
        return False

def _idx_map(n_kpts: int):
    """COCO-17 vs BlazePose-33 자동 대응"""
    if n_kpts >= 33:  # BlazePose/Mediapipe 계열
        return dict(LSh=11, RSh=12, LEl=13, REl=14, LWr=15, RWr=16,
                    LHp=23, RHp=24, LKn=25, RKn=26, LAn=27, RAn=28)
    else:             # COCO-17 계열
        return dict(LSh=5,  RSh=6,  LEl=7,  REl=8,  LWr=9,  RWr=10,
                    LHp=11, RHp=12, LKn=13, RKn=14, LAn=15, RAn=16)

def _hip_line_angle(hip: np.ndarray, knee: np.ndarray) -> Optional[float]:
    """
    허벅지 벡터(hip->knee)와 수평의 각(0~90). 실패 시 None.
    """
    try:
        dx, dy = float(knee[0] - hip[0]), float(knee[1] - hip[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        ang = math.degrees(math.atan2(abs(dy), abs(dx)))
        return ang if math.isfinite(ang) else None
    except Exception:
        return None

def compute_joint_angles(
    kxy: np.ndarray,
    kcf: np.ndarray,
    conf_thr: float = 0.2
) -> Dict[str, Optional[float]]:
    """
    입력:
      - kxy: (K,2) float32/64 좌표 (픽셀)
      - kcf: (K,) 각 키포인트 신뢰도
    출력:
      - 표기용 키: "Knee(L)", "Knee(R)", ..., "HipLine(R)"
        (값 없으면 None; NaN 사용 안 함)
    """
    kxy = np.asarray(kxy, dtype=np.float64)
    kcf = np.asarray(kcf, dtype=np.float64)
    if kxy.ndim != 2 or kxy.shape[1] != 2 or kcf.ndim != 1:
        raise ValueError("kxy shape must be (K,2) and kcf shape must be (K,)")

    idx = _idx_map(kxy.shape[0])
    LSh, RSh, LEl, REl, LWr, RWr, LHp, RHp, LKn, RKn, LAn, RAn = (
        idx["LSh"], idx["RSh"], idx["LEl"], idx["REl"], idx["LWr"], idx["RWr"],
        idx["LHp"], idx["RHp"], idx["LKn"], idx["RKn"], idx["LAn"], idx["RAn"]
    )

    ang: Dict[str, Optional[float]] = {}

    # 무릎
    ang["Knee(L)"] = _angle_deg(kxy[LHp], kxy[LKn], kxy[LAn]) if _ok(kxy, kcf, [LHp, LKn, LAn], conf_thr) else None
    ang["Knee(R)"] = _angle_deg(kxy[RHp], kxy[RKn], kxy[RAn]) if _ok(kxy, kcf, [RHp, RKn, RAn], conf_thr) else None
    # 힙
    ang["Hip(L)"]  = _angle_deg(kxy[LSh], kxy[LHp], kxy[LKn]) if _ok(kxy, kcf, [LSh, LHp, LKn], conf_thr) else None
    ang["Hip(R)"]  = _angle_deg(kxy[RSh], kxy[RHp], kxy[RKn]) if _ok(kxy, kcf, [RSh, RHp, RKn], conf_thr) else None
    # 어깨
    ang["Shoulder(L)"] = _angle_deg(kxy[LHp], kxy[LSh], kxy[LEl]) if _ok(kxy, kcf, [LHp, LSh, LEl], conf_thr) else None
    ang["Shoulder(R)"] = _angle_deg(kxy[RHp], kxy[RSh], kxy[REl]) if _ok(kxy, kcf, [RHp, RSh, REl], conf_thr) else None
    # 팔꿈치
    ang["Elbow(L)"] = _angle_deg(kxy[LSh], kxy[LEl], kxy[LWr]) if _ok(kxy, kcf, [LSh, LEl, LWr], conf_thr) else None
    ang["Elbow(R)"] = _angle_deg(kxy[RSh], kxy[REl], kxy[RWr]) if _ok(kxy, kcf, [RSh, REl, RWr], conf_thr) else None
    # 힙라인(허벅지 vs 수평)
    ang["HipLine(L)"] = _hip_line_angle(kxy[LHp], kxy[LKn]) if _ok(kxy, kcf, [LHp, LKn], conf_thr) else None
    ang["HipLine(R)"] = _hip_line_angle(kxy[RHp], kxy[RKn]) if _ok(kxy, kcf, [RHp, RKn], conf_thr) else None

    return ang

def update_meta_with_angles(
    meta: Dict,
    kxy: np.ndarray,
    kcf: np.ndarray,
    conf_thr: float = 0.2,
    ema: float = 0.0,
    prev: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    """
    compute_joint_angles() 결과를 기반으로:
      1) 좌우 평균 필드("Knee","Hip","Shoulder","Elbow","HipLine") 추가
      2) EMA 스무딩(선택)
      3) meta에 유효(유한수) 값만 동기화
      4) 스네이크 케이스 메타키도 함께 기록 (knee_l_deg 등)

    반환: 각도 dict (표기용 + 평균키), 값 없으면 None
    """
    ang = compute_joint_angles(kxy, kcf, conf_thr=conf_thr)

    # 좌/우 평균(가능한 값만)
    def _avg_pair(a_key: str, b_key: str) -> Optional[float]:
        a = ang.get(a_key, None)
        b = ang.get(b_key, None)
        if _is_finite_number(a) and _is_finite_number(b):
            return (float(a) + float(b)) / 2.0
        if _is_finite_number(a):
            return float(a)
        if _is_finite_number(b):
            return float(b)
        return None

    ang["Knee"]     = _avg_pair("Knee(L)", "Knee(R)")
    ang["Hip"]      = _avg_pair("Hip(L)", "Hip(R)")
    ang["Shoulder"] = _avg_pair("Shoulder(L)", "Shoulder(R)")
    ang["Elbow"]    = _avg_pair("Elbow(L)", "Elbow(R)")
    ang["HipLine"]  = _avg_pair("HipLine(L)", "HipLine(R)")

    # EMA 스무딩 (0 < ema < 1)
    if prev and (0.0 < ema < 1.0):
        smoothed: Dict[str, Optional[float]] = {}
        for k, v in ang.items():
            pv = prev.get(k) if isinstance(prev, dict) else None
            if not _is_finite_number(v) and not _is_finite_number(pv):
                smoothed[k] = None
            elif not _is_finite_number(v) and _is_finite_number(pv):
                smoothed[k] = float(pv)
            elif _is_finite_number(v) and not _is_finite_number(pv):
                smoothed[k] = float(v)
            else:
                smoothed[k] = float((1.0 - ema) * float(pv) + ema * float(v))
        ang = smoothed

    # meta에 동기화: 유효한 값만
    # 표기용 키 + 스네이크 케이스 키 모두 기록
    snake_map = {
        "Knee(L)": "knee_l_deg",       "Knee(R)": "knee_r_deg",
        "Hip(L)": "hip_l_deg",         "Hip(R)": "hip_r_deg",
        "Shoulder(L)": "shoulder_l_deg","Shoulder(R)": "shoulder_r_deg",
        "Elbow(L)": "elbow_l_deg",     "Elbow(R)": "elbow_r_deg",
        "HipLine(L)": "hipline_l_deg", "HipLine(R)": "hipline_r_deg",
        "Knee": "knee_avg_deg",        "Hip": "hip_avg_deg",
        "Shoulder": "shoulder_avg_deg","Elbow": "elbow_avg_deg",
        "HipLine": "hipline_avg_deg",
    }

    for k, v in ang.items():
        if _is_finite_number(v):
            meta[k] = float(v)
            if k in snake_map:
                meta[snake_map[k]] = float(v)

    return ang
