# views/summary_page.py 
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QFrame, QButtonGroup, QStackedLayout
)
from core.page_base import PageBase
import os

def pretty_hms(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def score_color(score: int) -> str:
    try:
        s = int(score)
    except Exception:
        return "#777"
    if s >= 90: return "#16a34a"
    if s >= 80: return "#2563eb"
    if s >= 70: return "#ea580c"
    return "#6b7280"

def asset_path(*parts) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, *parts)

# -------------------- MetricCard --------------------
class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "-", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("MetricTitle")
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("MetricValue")

        row = QVBoxLayout(self)
        row.setContentsMargins(18, 16, 18, 16)
        row.setSpacing(4)
        row.addWidget(self.title_lbl)
        row.addWidget(self.value_lbl)
        row.addStretch(1)

    def setValue(self, text: str):
        self.value_lbl.setText(text)

# -------------------- ExerciseCard (요약 모드) --------------------
class ExerciseCard(QFrame):
    def __init__(self, name: str = "-", reps: int | None = None, score: float | None = None,
                 placeholder: bool = False, placeholder_style: str = "dash",
                 parent: QWidget | None = None):
        """
        placeholder_style: "dash"  -> '-', '-', '-'
                           "blank" -> ''
        """
        super().__init__(parent)
        self.setObjectName("ExerciseCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.placeholder = placeholder
        self.placeholder_style = placeholder_style

        self.name_lbl = QLabel()
        self.name_lbl.setObjectName("ExName")
        self.count_value = QLabel()
        self.count_value.setObjectName("ExCount")
        self.score_value = QLabel()
        self.score_value.setObjectName("ExScore")

        # 레이아웃
        top = QHBoxLayout()
        top.addWidget(self.name_lbl)
        top.addStretch(1)

        cnt_box = self._pill_block("횟수", self.count_value)
        scr_box = self._pill_block("점수", self.score_value)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(cnt_box)
        bottom.addWidget(scr_box)
        bottom.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        lay.addLayout(top)
        lay.addLayout(bottom)

        # 초기 값 반영
        if placeholder:
            self.set_placeholder(placeholder_style)
        else:
            self.set_data(name, reps, score)

    def _pill_block(self, title: str, value_label: QLabel) -> QWidget:
        box = QFrame()
        box.setObjectName("PillBox")
        row = QHBoxLayout(box)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        t = QLabel(title)
        t.setObjectName("PillTitle")
        row.addWidget(t)
        row.addWidget(value_label)
        return box

    def set_placeholder(self, style: str = "dash"):
        self.placeholder = True
        self.placeholder_style = style
        if style == "blank":
            self.name_lbl.setText("")
            self.count_value.setText("")
            self.score_value.setText("")
        else:
            self.name_lbl.setText("-")
            self.count_value.setText("-")
            self.score_value.setText("-")
        self.setStyleSheet("QFrame#ExerciseCard { opacity: 0.4; }")

    def set_data(self, name: str, reps: int | None, score: float | None):
        self.placeholder = False
        self.setStyleSheet("")  # 투명도 복귀
        self.name_lbl.setText(name if name is not None else "-")
        r = int(reps or 0)
        s = int(round(score or 0))
        self.count_value.setText(f"{r}회")
        self.score_value.setText(f"{s}점")
        self.score_value.setStyleSheet(f"color: {score_color(s)};")

# -------------------- EMG (분석 모드) --------------------
class EmgPill(QFrame):
    """왼/오 다리 지표의 한 줄(타이틀 + 값) 표시"""
    def __init__(self, title: str, value: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("EmgPill")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(10)
        self._title = QLabel(title)
        self._title.setObjectName("EmgPillTitle")
        self._value = QLabel(value)
        self._value.setObjectName("EmgPillValue")
        row.addWidget(self._title)
        row.addStretch(1)
        row.addWidget(self._value)

    def set_value(self, value: str):
        self._value.setText(value)

class LegEmgCard(QFrame):
    """한쪽 다리(왼/오)의 근전도 요약 카드"""
    def __init__(self, side_title: str, data: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("LegCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.title = QLabel(side_title)
        self.title.setObjectName("LegTitle")

        self.pill_imb = EmgPill("불균형 지수", "+0%")
        self.pill_fat = EmgPill("피로도", "중간")
        self.pill_avg = EmgPill("평균 템포", "2.0 s/rep")
        self.pill_det = EmgPill("내림/올림", "1.1 s / 0.9 s")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        lay.addWidget(self.title)
        lay.addWidget(self.pill_imb)
        lay.addWidget(self.pill_fat)
        lay.addWidget(self.pill_avg)
        lay.addWidget(self.pill_det)
        lay.addStretch(1)

        self.set_data(data)

    def set_data(self, data: dict):
        imb = int(data.get("imbalance_pct", 0))
        fatigue = str(data.get("fatigue", "중간"))
        tempo = data.get("tempo", {})
        t_avg = str(tempo.get("avg", "2.0 s/rep"))
        t_down = str(tempo.get("down", "1.1 s"))
        t_up = str(tempo.get("up", "0.9 s"))

        self.pill_imb.set_value(f"{imb:+d}%")

        warn = abs(imb) >= 10
        self.pill_imb.setStyleSheet(
            "QFrame#EmgPill { }"  
        )
        self.pill_imb.findChild(QLabel, "EmgPillValue").setStyleSheet(
            f"color: {'#d32f2f' if warn else '#0f172a'}; font-weight: 800;"
        )
        self.pill_fat.set_value(fatigue)
        self.pill_avg.set_value(t_avg)
        self.pill_det.set_value(f"{t_down} / {t_up}")

class EmgPanel(QFrame):
    def __init__(self, emg_data: dict | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("EmgPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._defaults = {
            "left":  {"imbalance_pct": -12, "fatigue": "중간",
                      "tempo": {"avg": "2.1 s/rep", "down": "1.2 s", "up": "0.9 s"}},
            "right": {"imbalance_pct": 0, "fatigue": "낮음",
                      "tempo": {"avg": "2.1 s/rep", "down": "1.2 s", "up": "0.9 s"}},
            "note":  "양쪽 지수의 절댓값이 10% 이상이면 교정 권장"
        }

        d = emg_data or {}
        left_data  = d.get("left")  or self._defaults["left"]
        right_data = d.get("right") or self._defaults["right"]
        note_text  = d.get("note", self._defaults["note"])

        self.left_card  = LegEmgCard("왼쪽 다리", left_data)
        self.right_card = LegEmgCard("오른쪽 다리", right_data)
        self.tip = QLabel(note_text)
        self.tip.setObjectName("EmgTip")
        self.tip.setWordWrap(True)

        row = QHBoxLayout()
        row.setContentsMargins(18, 16, 18, 16)
        row.setSpacing(12)
        row.addWidget(self.left_card, 1)
        row.addWidget(self.right_card, 1)

        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(18, 16, 18, 16)
        wrap.setSpacing(8)
        wrap.addLayout(row)
        wrap.addWidget(self.tip)

    def set_data(self, emg_data: dict | None):
        d = emg_data or {}
        self.left_card.set_data(d.get("left")  or self._defaults["left"])
        self.right_card.set_data(d.get("right") or self._defaults["right"])
        self.tip.setText(d.get("note", self._defaults["note"]))

# -------------------- SummaryPage --------------------
class SummaryPage(PageBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SummaryPage")
        self._summary: dict = {}
        self.ctx = None
        self.mode = "summary"  

        # --- 배경 ---
        self._bg_pix = None
        self.bg = QLabel(self)
        self.bg.setScaledContents(False)
        self.bg.lower()
        bg_path = asset_path("assets", "background", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pm = QPixmap(bg_path)
            if not pm.isNull():
                self._bg_pix = pm
                self._rescale_bg()

        # --- 반투명 패널 ---
        self.panel = QFrame(self)
        self.panel.setObjectName("glassPanel")
        self.panel.setAttribute(Qt.WA_StyledBackground, True)

        root = QVBoxLayout(self.panel)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(18)

        # --- 상단바 ---
        top = QFrame()
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(22, 18, 22, 18)
        top_lay.setSpacing(14)

        self.title = QLabel("오늘의 운동 완료!")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        top_lay.addWidget(self.title, 1)

        self.btn_profile = QPushButton("내 정보")
        self.btn_profile.setObjectName("BtnUser")
        self.btn_profile.setFixedHeight(60)
        self.btn_profile.clicked.connect(self._on_profile)

        self.btn_retry = QPushButton("다시하기")
        self.btn_retry.setObjectName("BtnStart")
        self.btn_retry.setFixedHeight(60)
        self.btn_retry.clicked.connect(self._on_retry)

        top_lay.addWidget(self.btn_profile)
        top_lay.addWidget(self.btn_retry)

        # --- 상단 메트릭 카드 ---
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self.total_time_card = MetricCard("총 운동 시간", "00:00")
        self.total_reps_card = MetricCard("총 운동 횟수", "0회")
        self.avg_score_card = MetricCard("평균 점수", "0점")

        for w in (self.total_time_card, self.total_reps_card, self.avg_score_card):
            w.setMinimumHeight(90)
            metrics_row.addWidget(w)

        # --- 섹션 헤더: 제목 + 탭 버튼(요약/분석) ---
        header = QFrame()
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(12)

        section = QLabel("운동별 상세 결과")
        section.setObjectName("SectionTitle")

        header_lay.addWidget(section)
        header_lay.addStretch(1)

        self.tab_summary = QPushButton("요약")
        self.tab_summary.setObjectName("TabBtn")
        self.tab_summary.setCheckable(True)

        self.tab_analysis = QPushButton("분석")
        self.tab_analysis.setObjectName("TabBtn")
        self.tab_analysis.setCheckable(True)

        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_group.addButton(self.tab_summary)
        self.tab_group.addButton(self.tab_analysis)

        self.tab_summary.setChecked(True)
        self._apply_tab_style()

        self.tab_summary.toggled.connect(self._on_tab_changed)
        self.tab_analysis.toggled.connect(self._on_tab_changed)

        header_lay.addWidget(self.tab_summary)
        header_lay.addWidget(self.tab_analysis)

        # --- 콘텐츠 스택: 요약(그리드) / 분석(EMG) ---
        self.content_stack = QStackedLayout()
        # 1) 요약 컨테이너
        self.summary_container = QFrame()
        self.summary_grid = QGridLayout(self.summary_container)
        self.summary_grid.setHorizontalSpacing(12)
        self.summary_grid.setVerticalSpacing(12)
        self.exercise_cards: list[ExerciseCard] = []
        # 8개 고정 카드 생성(재사용)
        positions = [(r, c) for r in range(4) for c in range(2)]
        for (r, c) in positions:
            card = ExerciseCard(placeholder=True, placeholder_style="dash")
            self.exercise_cards.append(card)
            self.summary_grid.addWidget(card, r, c)

        # 2) 분석 컨테이너
        self.analysis_container = QFrame()
        analysis_lay = QVBoxLayout(self.analysis_container)
        analysis_lay.setContentsMargins(0, 0, 0, 0)
        analysis_lay.setSpacing(0)
        self.emg_panel = EmgPanel()  # 한 번만 만들고 재사용
        analysis_lay.addWidget(self.emg_panel)

        # 스택 구성
        self.content_stack.addWidget(self.summary_container)   # index 0
        self.content_stack.addWidget(self.analysis_container)  # index 1
        self.content_stack.setCurrentIndex(0)

        # --- 루트 레이아웃 조립 ---
        root.addWidget(top)
        root.addLayout(metrics_row)
        root.addWidget(header)
        root.addLayout(self.content_stack)

        self.setStyleSheet(self._stylesheet())

    def on_enter(self, ctx):
        self.ctx = ctx

    def set_data(self, summary: dict):
        # 원본 저장
        self._summary = dict(summary or {})
        # 상단 메트릭 업데이트
        d = self._summary
        total_seconds = d.get("duration_sec", 0)
        per_list = d.get("exercises") or []
        total_reps = sum(int(x.get("reps", 0)) for x in per_list)
        avg_score = 0
        if total_reps > 0:
            w_sum = sum(int(x.get("reps", 0)) * float(x.get("avg", x.get("avg_score", 0.0))) for x in per_list)
            avg_score = w_sum / total_reps
        self.total_time_card.setValue(pretty_hms(total_seconds))
        self.total_reps_card.setValue(f"{total_reps}회")
        self.avg_score_card.setValue(f"{int(round(avg_score))}점")

        # 요약 카드 업데이트 (기존 카드 재사용)
        self._update_summary_cards(per_list)
        # EMG 패널 업데이트 (기존 패널 재사용)
        self.emg_panel.set_data(d.get("emg"))

    # -------------------- 내부: 요약 카드 업데이트 --------------------
    def _update_summary_cards(self, per_list: list[dict]):
        # 최대 8칸, 부족분은 placeholder 유지
        for idx in range(8):
            card = self.exercise_cards[idx]
            if idx < len(per_list) and per_list[idx] is not None:
                data = per_list[idx]
                name = data.get("name", "-")
                reps = int(data.get("reps", 0))
                score = float(data.get("avg", data.get("avg_score", 0.0)))
                card.set_data(name, reps, score)
            else:
                card.set_placeholder("dash")

    # -------------------- 탭 / 버튼 핸들러 --------------------
    def _on_tab_changed(self, _checked: bool):
        self.mode = "summary" if self.tab_summary.isChecked() else "analysis"
        self._apply_tab_style()
        # 스택 전환만 (데이터는 이미 유지됨)
        self.content_stack.setCurrentIndex(0 if self.mode == "summary" else 1)

    def _apply_tab_style(self):
        self.tab_summary.setProperty("active", self.tab_summary.isChecked())
        self.tab_analysis.setProperty("active", self.tab_analysis.isChecked())
        self.tab_summary.style().unpolish(self.tab_summary); self.tab_summary.style().polish(self.tab_summary)
        self.tab_analysis.style().unpolish(self.tab_analysis); self.tab_analysis.style().polish(self.tab_analysis)

    # -------------------- Events --------------------
    def _on_retry(self):
        ex = (self._summary or {}).get("exercise")
        if self.ctx and hasattr(self.ctx, "restart_current_exercise"):
            self.ctx.restart_current_exercise(ex)

    def _on_profile(self):
        if self.ctx and hasattr(self.ctx, "goto_profile"):
            self.ctx.goto_profile()

    # -------------------- Resize & Layout --------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_bg()
        self._layout_panel()

    def _layout_panel(self):
        w, h = self.width(), self.height()
        target_w = max(int(w * 0.95), 1100)
        target_h = max(int(h * 0.95), 720)
        x = (w - target_w) // 2
        y = (h - target_h) // 2
        self.panel.setGeometry(x, y, target_w, target_h)

    def _rescale_bg(self):
        if self._bg_pix:
            self.bg.setGeometry(self.rect())
            scaled = self._bg_pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.bg.setPixmap(scaled)

    # -------------------- Stylesheet --------------------
    def _stylesheet(self) -> str:
        return """
        #glassPanel {
            background: rgba(255,255,255,1);
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.25);
        }
        #TopBar {
            background: rgba(138, 43, 226, 0.8);
            border-radius: 20px;
        }
        #Title {
            color: white;
            font-size: 44px;
            font-weight: 500;
            letter-spacing: 1px;
        }
        #BtnUser {
            background: rgba(40, 167, 69, 1);
            color: white;
            border: none;
            padding: 0 22px;
            border-radius: 14px;
            font-size: 24px;
            font-weight: 600;
        }
        #BtnUser:hover { background: rgba(50, 200, 85, 1); }

        #BtnStart {
            background: rgba(0, 123, 255, 1);
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 14px;
            font-size: 24px;
            font-weight: 700;
        }
        #BtnStart:hover { background: rgba(0, 105, 230, 1); }

        #MetricCard {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(52, 142, 219, 0.25), stop:1 rgba(52, 142, 219, 0.4));
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
        }
        #MetricTitle {
            color: #0b3b71;
            font-size: 30px;
            font-weight: 500;
        }
        #MetricValue {
            color: #0f172a;
            font-size: 32px;
            font-weight: 900;
        }
        #SectionTitle {
            color: rgba(0, 123, 255, 1);
            font-size: 60px;
            font-weight: 500;
            padding-left: 6px;
        }

        /* 탭 버튼 (파란색) */
        #TabBtn {
            background: #1976d2;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 12px;
            font-size: 22px;
            font-weight: 700;
        }
        #TabBtn:hover { background: #1565c0; }
        #TabBtn[active="true"] { background: #0d47a1; }

        #ExerciseCard {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 18px;
        }
        #ExName { font-size: 40px; font-weight: 500; color: #111827; }
        #ExCount { font-size: 25px; font-weight: 500; color: #0f172a; }
        #ExScore { font-size: 25px; font-weight: 500; }

        #PillBox {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eef7ff, stop:1 #e6f2ff);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 14px;
        }
        #PillTitle { color: #475569; font-size: 25px; font-weight: 500; }

        /* EMG 패널 */
        #EmgPanel {
            background: rgba(240, 249, 255, 0.8);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 18px;
        }
        #LegCard {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 18px;
        }
        #LegTitle {
            color: #0b3b71;
            font-size: 36px;
            font-weight: 600;
            margin-bottom: 6px;
        }
        #EmgPill {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eef7ff, stop:1 #e6f2ff);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 14px;
        }
        #EmgPillTitle { color: #475569; font-size: 24px; font-weight: 500; }
        #EmgPillValue { color: #0f172a; font-size: 26px; font-weight: 800; }
        #EmgTip { color: #475569; font-size: 18px; padding: 8px 4px; }
        """
