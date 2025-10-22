import time
import cv2
import sys
import os, subprocess, signal
import numpy as np
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import QTimer ,QUrl
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtMultimedia import QSoundEffect

from core.evaluators.pose_angles import update_meta_with_angles
from core.page_base import PageBase
from core.hailo_cam_adapter import HailoCamAdapter
from core.evaluators import get_evaluator_by_label, EvalResult, ExerciseEvaluator, get_advice_with_sfx  

from ui.score_painter import ScoreOverlay
from ui.overlay_painter import VideoCanvas, ExerciseCard, ScoreAdvicePanel, ActionButtons, AIMetricsPanel

PROJ_ROOT = Path("~/workspace/python/smart_gym_project/app").expanduser().resolve()

SERVICE_CMD_FIXED = (
    f'/bin/bash -lc "cd \'{PROJ_ROOT.as_posix()}\' && '
    'python3 sensor/squat_service_dual.py --user-seq --imu-master L --pair-lag-ms 980"'
)

MODEL_DIR = PROJ_ROOT / "sensor" / "models"
LOG_DIR  = PROJ_ROOT / "sensor" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SVC_LOG = (LOG_DIR / f"squat_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log").as_posix()
PRED_TSV = LOG_DIR / "reps_pred_dual.tsv"
IMU_TSV  = LOG_DIR / "imu_tempo.tsv"

_LABEL_KO = {
    None: "휴식중",
    "idle": "휴식중",
    "squat": "스쿼트",
    "pushup": "푸시업",
    "shoulder_press": "숄더프레스",
    "side_lateral_raise": "사레레",
    "bentover_dumbbell": "덤벨로우",
    "burpee": "버피",
    "leg_raise": "레그레이즈",
    "jumping_jack": "팔벌려뛰기",
}

EXERCISE_ORDER = [
    "squat",
    "pushup",
    "shoulder_press",
    "side_lateral_raise",
    "bentover_dumbbell",
    "leg_raise",
    "burpee",
    "jumping_jack",
]

class ExercisePage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("ExercisePage")

        self.cam = None
        self.state = "UP"
        self.reps = 0
        self._score_sum = 0.0
        self._score_n = 0
        self._session_started_ts = None
        self._last_label = None
        self._no_person_since = None
        self.NO_PERSON_TIMEOUT_SEC = 10.0
        self._entered_at = 0.0
        self.NO_PERSON_GRACE_SEC = 1.5

        self._active = False
        self._exercise_order: list[str] = list(EXERCISE_ORDER)
        self._per_stats: dict[str, dict] = {}

        self._svc_proc = None

        self.canvas = VideoCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        self.canvas.set_fit_mode("cover")

        self.card = ExerciseCard("휴식중")
        self.panel = ScoreAdvicePanel()
        self.panel.set_avg(0)
        self.panel.set_advice("올바른 자세로 준비하세요.")
        self.actions = ActionButtons()
        self.actions.endClicked.connect(self._end_clicked)
        self.actions.infoClicked.connect(self._info_clicked)

        self.ai_panel = AIMetricsPanel()
        self.score_overlay = ScoreOverlay(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.canvas, 1)

        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.PAGE_FPS_MS = 33

        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self._poll_tsv)
        self.AI_POLL_MS = 250

        self._title_hold = {"label": None, "cnt": 0}
        self._evaluator: ExerciseEvaluator | None = None
        self._last_eval_label: str | None = None

        self._last_pred_size = 0
        self._last_imu_size = 0

        self._angles_prev = None
        self._tempo_level_latest: str | None = None

        self._sfx_enabled = True
        self._sfx_dir = (PROJ_ROOT / "assets" / "advice").resolve()
        self._sfx_volume = 1
        self._SFX_COOLDOWN_SEC = 0.6
        self._last_sfx_ts = 0.0
        self._sfx_custom_map: dict[str, str] = {}

        self._sfx_player = QSoundEffect(self)
        self._sfx_player.setLoopCount(1)
        self._sfx_player.setVolume(self._sfx_volume)  # 0.0 ~ 1.0
        self._current_sfx_path = None

    def _init_per_stats(self):
        self._exercise_order = list(EXERCISE_ORDER)
        self._per_stats = {}
        for lb in self._exercise_order:
            name_ko = _LABEL_KO.get(lb, lb)
            self._per_stats[lb] = {
                "name": name_ko,
                "reps": 0,
                "score_sum": 0.0,
                "score_n": 0,
            }

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def _build_summary(self):
        per_list = []
        for lb in getattr(self, "_exercise_order", []): 
            ps = self._per_stats.get(lb) if hasattr(self, "_per_stats") else {}
            ps = ps or {}
            reps = int(ps.get("reps", 0))
            ssum = float(ps.get("score_sum", 0.0))
            sn   = int(ps.get("score_n", 0))
            avg  = (ssum / sn) if sn > 0 else 0.0
            per_list.append({
                "name": ps.get("name", _LABEL_KO.get(lb, lb)),
                "reps": reps,
                "avg": round(avg, 1),
            })

        w_sum    = sum(float(it["avg"]) * int(it["reps"]) for it in per_list)
        reps_sum = sum(int(it["reps"]) for it in per_list)
        avg_total = (w_sum / max(reps_sum, 1)) if reps_sum > 0 else 0.0

        ended_at = time.time()
        started_at = self._session_started_ts or ended_at
        duration_sec = int(max(0, ended_at - started_at))

        return {
            "duration_sec": duration_sec,
            "avg_score": round(avg_total, 1),
            "exercises": per_list,  
        }

    def _mount_overlays(self):
        self.canvas.clear_overlays()
        self.canvas.add_overlay(self.card, anchor="top-left")
        self.canvas.add_overlay(self.panel, anchor="top-right")
        self.canvas.add_overlay(self.actions, anchor="bottom-right")
        self.canvas.add_overlay(self.ai_panel, anchor="bottom-left")  
        self.card.show()
        self.panel.show()
        self.actions.show()
        self.ai_panel.show()
        self._sync_panel_sizes()

    def _sync_panel_sizes(self):
        W, H = self.width(), self.height()
        card_w = 650
        card_h = int(H * 0.5)
        panel_w = 600
        panel_h = int(H * 0.5)
        self.card.setFixedSize(card_w, card_h)
        self.panel.setFixedSize(panel_w, panel_h)

        pa_w = 600
        pa_h = int(H * 0.4)
        self.ai_panel.setFixedSize(pa_w, pa_h)

    def _start_service_if_needed(self):
        if self._svc_proc is not None and self._svc_proc.poll() is None:
            return

        pyexe = sys.executable  
        args = [
            pyexe,
            "sensor/squat_service_dual.py",
            "--user-seq",
            "--imu-master", "L",
            "--pair-lag-ms", "980",
        ]

        env = os.environ.copy()
        env["SGP_LOG_DIR"] = LOG_DIR.as_posix()

        try:
            self._svc_proc = subprocess.Popen(
                args,
                cwd=PROJ_ROOT.as_posix(),     
                stdout=open(SVC_LOG, "ab", buffering=0),
                stderr=subprocess.STDOUT, 
                env=env,
                start_new_session=True,
            )
            print(f"[ExercisePage] started service pid={self._svc_proc.pid} (log: {SVC_LOG})")
        except Exception as e:
            print(f"[ExercisePage] start service failed: {e}")
            self._svc_proc = None

    # ===== 클래스 내부: _stop_service 교체 =====
    def _stop_service(self):
        p = self._svc_proc
        self._svc_proc = None
        if not p:
            return

        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                p.wait(timeout=5)
        except Exception:
            pass

        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=2)
        except Exception:
            pass

        try:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        print("[ExercisePage] service stopped")

    def _poll_tsv(self):
        try:
            current_label = self._last_eval_label or "idle"
            show_values = (current_label == "squat")

            if not show_values:
                self.ai_panel.set_ai(fi_l=None, fi_r=None, stage_l=None, stage_r=None,
                                    bi=None, bi_stage=None, bi_text=None)
                self.ai_panel.set_imu(tempo_score=None, tempo_level=None, imu_state=None)
                return 

            if PRED_TSV.exists():
                sz = PRED_TSV.stat().st_size
                if sz != self._last_pred_size and sz > 0:
                    self._last_pred_size = sz
                    with PRED_TSV.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                    if len(lines) >= 2:
                        last = lines[-1].strip().split("\t")
                        fi_l     = float(last[3]) if len(last) > 3 else None
                        fi_r     = float(last[4]) if len(last) > 4 else None
                        bi       = float(last[8]) if len(last) > 8 else None
                        stage_l  = last[9]  if len(last) > 9  else None
                        stage_r  = last[10] if len(last) > 10 else None
                        bi_stage = last[11] if len(last) > 11 else None
                        bi_text  = last[12] if len(last) > 12 else None
                        self.ai_panel.set_ai(fi_l=fi_l, fi_r=fi_r, stage_l=stage_l, stage_r=stage_r,
                                             bi=bi, bi_stage=bi_stage, bi_text=bi_text)

            if IMU_TSV.exists():
                sz2 = IMU_TSV.stat().st_size
                if sz2 != self._last_imu_size and sz2 > 0:
                    self._last_imu_size = sz2
                    with IMU_TSV.open("r", encoding="utf-8") as f:
                        lines2 = f.readlines()
                    if len(lines2) >= 2:
                        last2 = lines2[-1].strip().split("\t")
                        imu_state   = last2[5] if len(last2) > 5 else None
                        tempo_score = int(float(last2[10])) if len(last2) > 10 else None
                        tempo_level = last2[11] if len(last2) > 11 else None
                        self.ai_panel.set_imu(tempo_score=tempo_score,
                                              tempo_level=tempo_level, imu_state=imu_state)
                        self._tempo_level_latest = tempo_level
        except Exception:
            pass  

    def _info_clicked(self):
        try:
            if hasattr(self.ctx, "goto_profile"):
                self.ctx.goto_profile()
        except Exception:
            pass

    # Lifecycle
    def on_enter(self, ctx):
        self.ctx = ctx
        self._session_started_ts = time.time()
        self._score_sum = 0.0
        self._score_n = 0
        self._reset_state()
        self._init_per_stats()

        self._evaluator = None
        self._last_eval_label = None
        self._no_person_since = None
        self._entered_at = time.time()

        title_text = getattr(self.ctx, "current_exercise", None) or "휴식중"
        self.card.set_title(title_text)

        self._mount_overlays()
        self.ai_panel.set_imu(tempo_score=None, tempo_level=None, imu_state=None)
        self.ai_panel.set_ai(fi_l=None, fi_r=None, stage_l=None, stage_r=None, bi=None, bi_stage=None, bi_text=None)

        try:
            self.ctx.face.stop_stream()
        except Exception:
            pass

        if not hasattr(self.ctx, "cam") or self.ctx.cam is None:
            self.ctx.cam = HailoCamAdapter()
        self.ctx.cam.start()
        print("[ExercisePage 임정민2] cam started")
        self._start_service_if_needed()
        
        
        self._active = True
        if self.timer.isActive():
            self.timer.stop()
        self.timer.start(self.PAGE_FPS_MS)

        self._last_pred_size = 0
        self._last_imu_size = 0

        if self.ai_timer.isActive():
            self.ai_timer.stop()
        self.ai_timer.start(self.AI_POLL_MS)

    def on_leave(self, ctx):
        self._active = False
        if self.timer.isActive(): self.timer.stop()
        if self.ai_timer.isActive(): self.ai_timer.stop()
        try:
            ctx.cam.stop()
        except Exception:
            pass

        self._stop_service()

        self.canvas.clear_overlays()
        self._evaluator = None
        self._last_eval_label = None

    # End Button
    def _end_clicked(self):
        self._active = False
        if self.timer.isActive(): self.timer.stop()
        if self.ai_timer.isActive(): self.ai_timer.stop()
        try:
            self.ctx.cam.stop()
        except Exception:
            pass

        self._stop_service()

        summary = self._build_summary()
        try:
            if hasattr(self.ctx, "save_workout_session"): self.ctx.save_workout_session(summary)
        except Exception:
            pass
        if hasattr(self.ctx, "goto_summary"): self.ctx.goto_summary(summary)
        self.canvas.clear_overlays()

    # Tick (기존 포즈 & 점수 로직 유지)
    def _tick(self):
        if not self._active or not self.timer.isActive():
            return

        meta = self.ctx.cam.meta() or {}
        now = time.time()
        in_grace = (now - self._entered_at) < self.NO_PERSON_GRACE_SEC

        m_ok = bool(meta.get("ok", False))
        if in_grace:
            self._no_person_since = None
        else:
            if not m_ok:
                if self._no_person_since is None:
                    self._no_person_since = now
                elif (now - self._no_person_since) >= self.NO_PERSON_TIMEOUT_SEC:
                    self._active = False
                    try:
                        if self.timer.isActive(): self.timer.stop()
                    except Exception:
                        pass
                    try:
                        self.ctx.cam.stop()
                    except Exception:
                        pass
                    self._stop_service()
                    self._no_person_since = None
                    self._goto("guide")
                    return
            else:
                self._no_person_since = None

        frame = self.ctx.cam.frame()
        if frame is not None:
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                people = self.ctx.cam.people()
                EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]
                if people:
                    H, W = bgr.shape[:2]
                    max_len2 = (max(W, H)*0.6) ** 2
                    LINE_COLOR = (144, 238, 144)
                    for p in people:
                        pts = p.get("kpt", [])
                        vis = [len(pt)>=3 and float(pt[2])>=0.65 for pt in pts]
                        for a,b in EDGES:
                            if a<len(pts) and b<len(pts) and vis[a] and vis[b]:
                                x1_,y1_ = int(pts[a][0]), int(pts[a][1])
                                x2_,y2_ = int(pts[b][0]), int(pts[b][1])
                                dx,dy = x1_-x2_, y1_-y2_
                                if (dx*dx + dy*dy) <= max_len2:
                                    cv2.line(bgr, (x1_,y1_), (x2_,y2_), LINE_COLOR, 2)

                try:
                    if people:
                        kpt = people[0].get("kpt", [])
                        if kpt and len(kpt) >= 17:
                            kxy = np.array([[pt[0], pt[1]] for pt in kpt], dtype=np.float32)
                            kcf = np.array([(pt[2] if len(pt) > 2 else 1.0) for pt in kpt], dtype=np.float32)
                            angles = update_meta_with_angles(
                                meta, kxy, kcf, conf_thr=0.5, ema=0.2,
                                prev=getattr(self, "_angles_prev", None),
                            )
                            self._angles_prev = angles
                            meta["_kpt"] = kpt
                except Exception:
                    pass

                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                if self._active:
                    self.canvas.set_frame(qimg)
            except cv2.error:
                return

        raw_label = meta.get("label", None)
        title_kor = _LABEL_KO.get(raw_label, (raw_label if raw_label else "휴식중"))
        hold = self._title_hold
        if hold["label"] != title_kor:
            hold["label"] = title_kor
            hold["cnt"] = 1
        else:
            hold["cnt"] += 1
        if hold["cnt"] >= 3 and title_kor != self._last_label:
            self.card.set_title(title_kor)
            self._last_label = title_kor

        label = raw_label if raw_label else "idle"
        if self._last_eval_label != label:
            self._last_eval_label = label
            self._evaluator = get_evaluator_by_label(label) if label not in (None, "idle") else None
            if self._evaluator: 
                self._evaluator.reset()

            if label != "squat":
                self.ai_panel.set_ai(fi_l=None, fi_r=None, stage_l=None, stage_r=None,
                                    bi=None, bi_stage=None, bi_text=None)
                self.ai_panel.set_imu(tempo_score=None, tempo_level=None, imu_state=None)
                self._tempo_level_latest = None

        if label in (None, "idle") or not self._evaluator:
            self.panel.set_advice("올바른 자세로 준비하세요.")
            return

        try:
            res: EvalResult = self._evaluator.update(meta)
        except Exception:
            return
        if not res:
            return

        if res.advice:
            self.panel.set_advice(res.advice)
             # --- 점수에 따른 SFX 재생 (선택) ---
        if res.score is not None:
            try:
                _text_unused, _bucket, sfx_key = get_advice_with_sfx(label, int(res.score), ctx=None)
                self._play_sfx(sfx_key)
            except Exception as e:
                print("[SFX] error:", e)
        if res.rep_inc:
            self.reps += res.rep_inc
            if hasattr(self.card, "set_count"):
                self.card.set_count(self.reps)
            elif hasattr(self.card, "set_reps"):
                self.card.set_reps(self.reps)

            ps = self._per_stats.get(label)
            if ps is not None:
                ps["reps"] = int(ps.get("reps", 0)) + int(res.rep_inc)

        if res.score is not None:
            s = int(res.score)
            if s >= 80:
                color = QColor(0, 128, 255)     # 파란색
            elif s >= 60:
                color = QColor(0, 200, 0)       # 초록색
            elif s >= 40:
                color = QColor(255, 140, 0)     # 주황색
            else:
                color = QColor(255, 0, 0)       # 빨강  
            
            supertext = None
            super_color = None
            if label == "squat":
                raw_lvl = (self._tempo_level_latest or "").strip()
                lvl_up = raw_lvl.upper()

                mapping_text = {
                    "A_매우안정": "PERFECT!", "A": "PERFECT!",
                    "B_안정": "GOOD",        "B": "GOOD",
                    "C_보통": "NOT BAD",     "C": "NOT BAD",
                    "D_불안정": "BAD",       "D": "BAD",
                }
                if lvl_up in mapping_text:
                    supertext = f"TEMPO {mapping_text[lvl_up]}"
                    if   lvl_up in ("A", "A_매우안정"):
                        super_color = QColor(0, 128, 255)   # 매우안정 → 파랑
                    elif lvl_up in ("B", "B_안정"):
                        super_color = QColor(0, 200, 0)     # 안정 → 초록
                    elif lvl_up in ("C", "C_보통"):
                        super_color = QColor(255, 140, 0)   # 보통 → 주황
                    elif lvl_up in ("D", "D_불안정"):
                        super_color = QColor(255, 0, 0)     # 불안정 → 빨강
                elif raw_lvl:
                    supertext = f"TEMPO UNKNOWN"

            self.score_overlay.show_score(
                str(s),
                text_qcolor=color,              # 점수 색 (그대로)
                supertext=supertext,            # 템포 문구
                super_qcolor=super_color        # 템포 글자색만 별도 지정
            )

            self._score_sum += float(res.score)
            self._score_n += 1
            avg = round(self._score_sum / max(1, self._score_n), 1)
            self.panel.set_avg(avg)

            ps = self._per_stats.get(label)
            if ps is not None:
                ps["score_sum"] = float(ps.get("score_sum", 0.0)) + float(res.score)
                ps["score_n"]   = int(ps.get("score_n", 0)) + 1

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_panel_sizes()
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

    # Reset State
    def _reset_state(self):
        self.state = "UP"
        self.reps = 0
        self.card.set_count(0)
        self.panel.set_avg(0)
        self.panel.set_advice("올바른 자세로 준비하세요.")
        self._tempo_level_latest = None 
        if self._evaluator:
            self._evaluator.reset()

        self.ai_panel.set_imu(tempo_score=None, tempo_level=None, imu_state=None)
        self.ai_panel.set_ai(fi_l=None, fi_r=None, stage_l=None, stage_r=None, bi=None, bi_stage=None, bi_text=None)

    def _resolve_sfx_path(self, sfx_key: str) -> str | None:
        print(f"[음성파일  실행여부]")
        if not sfx_key:
            return None
        # 1) 사용자 지정 매핑 우선
        if sfx_key in self._sfx_custom_map:
            p = Path(self._sfx_custom_map[sfx_key]).expanduser().resolve()
            return p.as_posix() if p.exists() else None
        # 2) 기본 규칙: app/assets/advice/{sfx_key}.wav
        cand = self._sfx_dir / f"{sfx_key}.wav"
        return cand.as_posix() if cand.exists() else None

    def _play_sfx(self, sfx_key: str):
        if not self._sfx_enabled or not sfx_key:
            return

        now = time.time()
        if (now - self._last_sfx_ts) < self._SFX_COOLDOWN_SEC:
            return

        # 재생 중이면 절대 새 소리 안 틀기
        try:
            if self._sfx_player.isPlaying():
                return
        except Exception:
            pass

        path = self._resolve_sfx_path(sfx_key)
        if not path:
            return

        if self._current_sfx_path != path:
            self._sfx_player.setSource(QUrl.fromLocalFile(path))
            self._current_sfx_path = path

        self._sfx_player.play()
        self._last_sfx_ts = now
