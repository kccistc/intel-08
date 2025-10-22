import traceback
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import func
from db.models import User, WorkoutSession, SessionExercise

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
    QGridLayout, QFrame, QProgressBar, QSizePolicy, QScrollArea, QStackedLayout, QButtonGroup
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QPainter, QBrush

from core.page_base import PageBase
import pyqtgraph as pg

from ui.info_style import apply_info_page_styles

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

class Avatar(QWidget):
    """원형 배경 위에 이니셜 한 글자."""
    def __init__(self, name="?", size=56, color="#7c8cf8", parent=None):
        super().__init__(parent)
        self.name = name
        self.size_px = size
        self.color = QColor(color)
        self.setFixedSize(QSize(size, size))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self.color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(self.rect())
        init = (self.name[:1] if self.name else "?")
        p.setPen(QColor("white"))
        f = QFont("", int(self.size_px * 0.42), QFont.Bold)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter, init)

class _NoMouseViewBox(pg.ViewBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setMenuEnabled(False)

    def mouseClickEvent(self, ev):   ev.ignore()
    def mouseDragEvent(self, ev):    ev.ignore()
    def wheelEvent(self, ev):        ev.ignore()
    def contextMenuEvent(self, ev):  ev.ignore()

class _HBar(QFrame):
    def __init__(self, title: str, color="#1976d2", parent=None):
        super().__init__(parent)
        self.setObjectName("_HBar")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.title = QLabel(title)
        self.title.setProperty("cls", "muted")
        self.value = QLabel("0%")
        self.value.setProperty("cls", "title")
        row.addWidget(self.title)
        row.addStretch(1)
        row.addWidget(self.value)
        lay.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background: #e8eef7; border: 0; border-radius: 8px; height: 14px;
            }}
            QProgressBar::chunk {{
                background: {color}; border-radius: 8px;
            }}
        """)
        lay.addWidget(self.bar)

    def set_percent(self, p: int):
        p = max(0, min(100, int(p)))
        self.value.setText(f"{p}%")
        self.bar.setValue(p)


class _AnalysisPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AnalysisPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(10)

        # 상단 KPI 두 개 (오늘 스쿼트)
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(18)

        self.kpi_squat = QFrame()
        ks = QVBoxLayout(self.kpi_squat)
        ks.setContentsMargins(14, 12, 14, 12)
        ks.setSpacing(2)
        ttl1 = QLabel("오늘 스쿼트")
        ttl1.setProperty("cls", "muted")
        self.lbl_squat_cnt = QLabel("0회")
        self.lbl_squat_cnt.setProperty("cls", "kpi")
        ks.addWidget(ttl1)
        ks.addWidget(self.lbl_squat_cnt)

        self.kpi_avg = QFrame()
        ka = QVBoxLayout(self.kpi_avg)
        ka.setContentsMargins(14, 12, 14, 12)
        ka.setSpacing(2)
        ttl2 = QLabel("평균 점수")
        ttl2.setProperty("cls", "muted")
        self.lbl_avg_score = QLabel("0점")
        self.lbl_avg_score.setProperty("cls", "kpi")
        ka.addWidget(ttl2)
        ka.addWidget(self.lbl_avg_score)

        kpi_row.addWidget(self.kpi_squat, 1)
        kpi_row.addWidget(self.kpi_avg, 1)
        lay.addLayout(kpi_row)

        # 불균형
        cap1 = QLabel("왼/오른쪽 다리 불균형")
        cap1.setProperty("cls", "title")
        lay.addWidget(cap1)

        self.bar_left  = _HBar("왼쪽",  "#0ea5e9")  
        self.bar_right = _HBar("오른쪽", "#34d399") 
        lay.addWidget(self.bar_left)
        lay.addWidget(self.bar_right)

        # 템포
        cap2 = QLabel("템포 비율")
        cap2.setProperty("cls", "title")
        lay.addWidget(cap2)

        self.bar_down = _HBar("내림", "#6366f1")   # 보라
        self.bar_up   = _HBar("올림", "#f59e0b")   # 주황
        lay.addWidget(self.bar_down)
        lay.addWidget(self.bar_up)
        lay.addStretch(1)

        # 약간의 카드형 배경
        self.setStyleSheet("""
            QFrame#AnalysisPanel {
                background: #ffffff;
                border: 1px solid rgba(0,0,0,0.06);
                border-radius: 12px;
            }
            QLabel[cls="kpi"] {
                font-size: 32px;
                font-weight: 900;
                color: #0f172a;
            }
            QLabel[cls="muted"] {
                color: #6b7280;
                font-size: 18px;
                font-weight: 600;
            }
            QLabel[cls="title"] {
                color: #0b3b71;
                font-size: 22px;
                font-weight: 700;
                margin-top: 4px;
            }
        """)

    def set_data(self, data: dict):
        """
        data = {
          "today":   {"squat_reps": int, "squat_avg": float},
          "imb":     {"left": int, "right": int},   
          "tempo":   {"down": int, "up": int}       
        }
        """
        today = (data or {}).get("today", {})
        self.lbl_squat_cnt.setText(f"{int(today.get('squat_reps', 0))}회")
        self.lbl_avg_score.setText(f"{int(round(float(today.get('squat_avg', 0))))}점")

        imb = (data or {}).get("imb", {})
        l = int(imb.get("left", 50)); r = int(imb.get("right", 50))
        s = max(1, l + r)
        l = round(l * 100 / s); r = 100 - l
        self.bar_left.set_percent(l)
        self.bar_right.set_percent(r)

        tempo = (data or {}).get("tempo", {})
        d = int(tempo.get("down", 50)); u = int(tempo.get("up", 50))
        s2 = max(1, d + u)
        d = round(d * 100 / s2); u = 100 - d
        self.bar_down.set_percent(d)
        self.bar_up.set_percent(u)

class InfoPage(PageBase):
    BASE_COLORS = [
        "#6aa7ff", "#9b6bff", "#19c37d", "#f59e0b", "#ef4444",
        "#22c55e", "#06b6d4", "#f472b6", "#a3e635", "#fb7185",
        "#f97316", "#60a5fa", "#a78bfa", "#34d399", "#4ade80"
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("InfoPage")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#f5f8ff"))
        self.setPalette(pal)

        pg.setConfigOptions(antialias=True)

        self._ex_color = {}
        self._stat_cards = {}

        self._build_ui()

    def _build_ui(self):
        apply_info_page_styles(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_bar.setStyleSheet("""
            QFrame#TopBar {
                background: rgba(126, 58, 242, 0.92); /* 보라색 */
                border-radius:12px;
            }
            QLabel#TopTitle {
                color: #ffffff;
                font-size: 40px; /* 큰 타이틀 */
                font-weight: 700;
                padding-left: 4px;
            }
            QPushButton#BtnPrimary {
                background:#2563eb; color:white;
                border:1px solid #1d4ed8;
                border-radius:10px; padding:10px 18px;
                font-size:25px; font-weight:500;
            }
            QPushButton#BtnPrimary:hover { background:#1d4ed8; }
            QPushButton#BtnDanger {
                background:#ef4444; color:white;
                border:1px solid #dc2626;
                border-radius:10px; padding:10px 18px;
                font-size:25px; font-weight:500;
            }
            QPushButton#BtnDanger:hover { background:#dc2626; }
        """)
        tb = QHBoxLayout(top_bar)
        tb.setContentsMargins(16, 10, 16, 10)
        # 좌측 타이틀
        self.top_title = QLabel("회원 정보")
        self.top_title.setObjectName("TopTitle")
        tb.addWidget(self.top_title, 0, Qt.AlignVCenter)
        tb.addStretch(1)
        # 우측 버튼들
        self.btn_back = QPushButton("운동 화면으로")
        self.btn_back.setObjectName("BtnPrimary")
        self.btn_logout = QPushButton("로그아웃")
        self.btn_logout.setObjectName("BtnDanger")
        tb.addWidget(self.btn_back)
        tb.addWidget(self.btn_logout)
        root.addWidget(top_bar)

        # 그리드
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 3)

        # 좌측 프로필 카드
        self.profile_card = Card()
        self.profile_card.setMaximumWidth(360)
        p = QVBoxLayout(self.profile_card)
        p.setContentsMargins(18, 16, 18, 16)
        p.setSpacing(10)

        # 상단: 아바타 + 이름(크게)
        row = QHBoxLayout()
        self.avatar = Avatar("김", size=64)
        row.addWidget(self.avatar)

        namebox = QVBoxLayout()
        self.lbl_name = QLabel("—")
        self.lbl_name.setProperty("cls", "display")
        self.lbl_grade = QLabel("정회원")
        self.lbl_grade.setProperty("cls", "muted")
        namebox.addWidget(self.lbl_name)
        namebox.addWidget(self.lbl_grade)
        row.addLayout(namebox)
        p.addLayout(row)

        # 가입/만료/마지막 운동일(제목-날짜)
        self.lbl_join_cap = QLabel("가입일")
        self.lbl_join_cap.setProperty("cls", "muted")
        self.lbl_join_date = QLabel("—")
        self.lbl_join_date.setProperty("cls", "date")

        self.lbl_until_cap = QLabel("회원권 만료일")
        self.lbl_until_cap.setProperty("cls", "muted")
        self.lbl_until_date = QLabel("—")
        self.lbl_until_date.setProperty("cls", "date")

        self.lbl_last_workout_cap = QLabel("마지막 운동일")
        self.lbl_last_workout_cap.setProperty("cls", "muted")
        self.lbl_last_workout = QLabel("—")
        self.lbl_last_workout.setProperty("cls", "date")

        p.addWidget(self.lbl_join_cap)
        p.addWidget(self.lbl_join_date)
        p.addSpacing(4)
        p.addWidget(self.lbl_until_cap)
        p.addWidget(self.lbl_until_date)
        p.addSpacing(4)
        p.addWidget(self.lbl_last_workout_cap)
        p.addWidget(self.lbl_last_workout)

        # 남은 기간 + 진행바
        self.lbl_days_left = QLabel("남은 기간 —일")
        p.addWidget(self.lbl_days_left)

        self.pb_days = QProgressBar()
        self.pb_days.setTextVisible(False)
        self.pb_days.setFixedHeight(12)
        p.addWidget(self.pb_days)

        # 주간 합계
        self.lbl_week_total = QLabel("이번 주 총 운동 횟수 0회")
        self.lbl_week_total.setProperty("cls", "title")
        p.addWidget(self.lbl_week_total)

        grid.addWidget(self.profile_card, 0, 0, 2, 1)

        # 우상단 통계 카드 리스트
        right_top = Card()
        rt = QVBoxLayout(right_top)
        rt.setContentsMargins(12, 12, 12, 8)
        rt.setSpacing(8)

        nav = QHBoxLayout()
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        for b in (self.btn_left, self.btn_right):
            b.setFixedWidth(48)          
            b.setMinimumHeight(32)       
            b.setStyleSheet("font-size:20px; padding:4px 8px;")
        nav.addStretch(1)
        nav.addWidget(self.btn_left)
        nav.addWidget(self.btn_right)
        rt.addLayout(nav)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{border:0; background:transparent;} QWidget{background:transparent;}")

        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(6, 6, 6, 6)
        self.cards_layout.setSpacing(12)
        self.scroll.setWidget(self.cards_container)
        rt.addWidget(self.scroll)

        grid.addWidget(right_top, 0, 1, 1, 3)

        # 하단 차트 카드
        self.chart_card = Card()
        self.chart_card.setStyleSheet("""
            QPushButton#TabBtn {
                background: #1976d2;
                color: white;
                border: none;
                padding: 8px 14px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton#TabBtn:hover { background: #1565c0; }
            QPushButton#TabBtn[active="true"] { background: #0d47a1; }

            QFrame#ChartBlank {
                background: #ffffff;
                border: 1px dashed rgba(0,0,0,0.06);
                border-radius: 8px;
                min-height: 260px;  /* 카드 높이 유지용, 필요시 조정 */
            }
        """)
        cc = QVBoxLayout(self.chart_card)
        cc.setContentsMargins(16, 14, 16, 12)
        cc.setSpacing(6)

        title_row = QHBoxLayout()
        icon = QLabel("▴")
        icon.setProperty("cls", "icon")
        t = QLabel("주간 운동 추이")
        t.setProperty("cls", "title")
        title_row.addWidget(icon)
        title_row.addWidget(t)
        title_row.addStretch(1)

        # 탭 버튼 (요약/분석)
        self.chart_tab_summary = QPushButton("요약")
        self.chart_tab_summary.setObjectName("TabBtn")
        self.chart_tab_summary.setCheckable(True)

        self.chart_tab_analysis = QPushButton("분석")
        self.chart_tab_analysis.setObjectName("TabBtn")
        self.chart_tab_analysis.setCheckable(True)

        self.chart_tab_group = QButtonGroup(self)
        self.chart_tab_group.setExclusive(True)
        self.chart_tab_group.addButton(self.chart_tab_summary)
        self.chart_tab_group.addButton(self.chart_tab_analysis)

        self.chart_tab_summary.setChecked(True)
        self.chart_tab_summary.toggled.connect(self._on_chart_tab_changed)
        self.chart_tab_analysis.toggled.connect(self._on_chart_tab_changed)

        title_row.addWidget(self.chart_tab_summary)
        title_row.addWidget(self.chart_tab_analysis)

        cc.addLayout(title_row)

        self.chart_stack = QStackedLayout()

        self.vb = _NoMouseViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb, background="#ffffff")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().setMenuEnabled(False)
        ax = self.plot.getAxis('bottom'); ax.setPen(pg.mkPen('#9fb0c6'))
        ax = self.plot.getAxis('left');   ax.setPen(pg.mkPen('#9fb0c6'))

        # 차트 라벨: 20pt(≈26~27px)
        self.plot.setLabel('left', "횟수", **{'color': '#5b6a82', 'size': '20pt'})
        self.plot.setLabel('bottom', "요일", **{'color': '#5b6a82', 'size': '20pt'})

        # 눈금 폰트
        tick_font = QFont()
        tick_font.setPointSize(18)        # ≈ 24px
        tick_font.setWeight(QFont.Medium) # 500
        self.plot.getAxis('bottom').setStyle(tickFont=tick_font)
        self.plot.getAxis('left').setStyle(tickFont=tick_font)

        summary_wrap = QFrame()
        _sw_lay = QVBoxLayout(summary_wrap)
        _sw_lay.setContentsMargins(0, 0, 0, 0)
        _sw_lay.setSpacing(0)
        _sw_lay.addWidget(self.plot, 1)

        self.chart_blank = QFrame()
        self.chart_blank.setObjectName("ChartBlank")
        _bk_lay = QVBoxLayout(self.chart_blank)
        _bk_lay.setContentsMargins(0, 0, 0, 0)
        _bk_lay.setSpacing(0)

        self.analysis_panel = _AnalysisPanel()
        _bk_lay.addWidget(self.analysis_panel)

        self.chart_stack.addWidget(summary_wrap)   
        self.chart_stack.addWidget(self.chart_blank)  
        self.chart_stack.setCurrentIndex(0)

        cc.addLayout(self.chart_stack)

        grid.addWidget(self.chart_card, 1, 1, 1, 3)
        root.addLayout(grid)

        self.btn_back.clicked.connect(lambda: self._goto("guide"))
        self.btn_logout.clicked.connect(self._logout)
        self.btn_left.clicked.connect(lambda: self._scroll_stats(-1))
        self.btn_right.clicked.connect(lambda: self._scroll_stats(+1))
        self._apply_chart_tab_style()

    def on_enter(self, ctx):
        self.ctx = ctx
        self._refresh()

    def _refresh(self):
        if not self.ctx.is_logged_in():
            self._show_logged_out_view()
            return
        try:
            with self.ctx.SessionLocal() as s:
                user = s.query(User).filter_by(id=self.ctx.current_user_id).one_or_none()
                if not user:
                    self.ctx.clear_current_user()
                    self._show_logged_out_view()
                    return

                # 프로필 텍스트
                self.lbl_name.setText(f"{user.name}")
                if hasattr(self, "avatar"):
                    self.avatar.name = (user.name or "?")
                    self.avatar.update()

                # 가입일 / 만료일 / 마지막 운동일
                self.lbl_join_date.setText(self._fmt_date(getattr(user, 'created_at', None)))

                today = datetime.now().date()
                mock_until = today + timedelta(days=54)
                self.lbl_until_date.setText(mock_until.strftime('%Y년 %m월 %d일'))

                # 마지막 운동일(최근 세션)
                last_row = (
                    s.query(func.max(WorkoutSession.started_at))
                    .filter(WorkoutSession.user_id == user.id)
                    .one()
                )
                last_dt = last_row[0] if last_row and last_row[0] else None
                self.lbl_last_workout.setText(self._fmt_date(last_dt))

                # 남은 일수/진행바
                days_left = (mock_until - today).days
                self.lbl_days_left.setText(f"남은 기간  {days_left}일")
                self.pb_days.setMaximum(days_left if days_left > 0 else 1)
                self.pb_days.setValue(max(0, days_left))

                # ── 7일치 집계(실제 데이터만 표시) ───────────────────────────
                start_date = today - timedelta(days=6)
                date_expr = func.date(WorkoutSession.started_at)
                rows = (
                    s.query(
                        date_expr.label("d"),
                        SessionExercise.exercise_name.label("ex"),
                        func.sum(SessionExercise.reps).label("cnt"),
                        func.avg(SessionExercise.avg_score).label("avg"),
                    )
                    .join(SessionExercise, SessionExercise.session_id == WorkoutSession.id)
                    .filter(WorkoutSession.user_id == user.id)
                    .filter(WorkoutSession.started_at.isnot(None))
                    .filter(date_expr >= str(start_date))
                    .group_by("d", "ex")
                    .all()
                )

                days = [(start_date + timedelta(days=i)).isoformat() for i in range(7)]
                by_day = defaultdict(lambda: defaultdict(int))
                for d, ex, cnt, _ in rows:
                    by_day[str(d)][ex] = int(cnt or 0)

                week_total = sum(sum(by_day[d].values()) for d in days)
                self.lbl_week_total.setText(f"총 운동 횟수  {week_total:,}회")

                stats = (
                    s.query(
                        SessionExercise.exercise_name.label("ex"),
                        func.sum(SessionExercise.reps).label("total"),
                        func.avg(SessionExercise.avg_score).label("avg"),
                    )
                    .join(WorkoutSession, SessionExercise.session_id == WorkoutSession.id)
                    .filter(WorkoutSession.user_id == user.id)
                    .group_by("ex")
                    .all()
                )
                stat_map = {ex: (int(total or 0), float(avg or 0)) for ex, total, avg in stats}

                self._ensure_colors(list(stat_map.keys()))
                self._rebuild_stat_cards(stat_map)

                date_expr = func.date(WorkoutSession.started_at)
                today_str = str(today)
                row_today = (
                    s.query(
                        func.sum(SessionExercise.reps).label("cnt"),
                        func.avg(SessionExercise.avg_score).label("avg"),
                    )
                    .join(WorkoutSession, SessionExercise.session_id == WorkoutSession.id)
                    .filter(WorkoutSession.user_id == user.id)
                    .filter(SessionExercise.exercise_name == "squat")
                    .filter(date_expr == today_str)
                    .one()
                )
                today_cnt = int(row_today[0] or 0)
                today_avg = float(row_today[1] or 0.0)

                analysis_data = {
                    "today": {"squat_reps": today_cnt, "squat_avg": today_avg},
                    "imb":   {"left": 54, "right": 46},     
                    "tempo": {"down": 58, "up": 42},         
                }
                if hasattr(self, "analysis_panel") and self.analysis_panel:
                    self.analysis_panel.set_data(analysis_data)

                self._render_line_chart(days, by_day)

        except Exception:
            traceback.print_exc()
            QMessageBox.critical(self, "오류", "정보를 불러오지 못했습니다.")
            self._show_profile_only()

    def _rebuild_stat_cards(self, stat_map):
        # 초기화
        for i in reversed(range(self.cards_layout.count())):
            w = self.cards_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self._stat_cards.clear()

        # 카드 생성(총 횟수 내림차순)
        for ex, (total, avg) in sorted(stat_map.items(), key=lambda x: -x[1][0]):
            color = self._ex_color.get(ex, "#6aa7ff")
            card = self._make_mini_card(ex, color, total, avg)
            self.cards_layout.addWidget(card)
            self._stat_cards[ex] = card

        self.cards_layout.addStretch(1)

    def _make_mini_card(self, ex, color, total, avg):
        w = Card()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)

        row = QHBoxLayout()
        dot = QLabel("●")
        dot.setProperty("cls", "icon")
        dot.setStyleSheet(f"color:{color};")
        ttl = QLabel(self._ex_display(ex))
        ttl.setProperty("cls", "muted")

        row.addWidget(dot)
        row.addWidget(ttl)
        row.addStretch(1)
        lay.addLayout(row)

        total_lbl = QLabel(f"{total:,}회")
        total_lbl.setProperty("cls", "kpi")

        avg_lbl = QLabel(f"평균 점수  {avg:.0f} 점")
        avg_lbl.setProperty("cls", "muted")

        lay.addWidget(total_lbl)
        lay.addWidget(avg_lbl)
        lay.addStretch(1)

        w.setFixedWidth(220)
        return w

    def _scroll_stats(self, direction: int):
        bar = self.scroll.horizontalScrollBar()
        step = int(self.scroll.viewport().width() * 0.8)
        bar.setValue(max(0, bar.value() + (step * direction)))

    def _on_chart_tab_changed(self, _checked: bool):
        is_summary = self.chart_tab_summary.isChecked()
        self.chart_stack.setCurrentIndex(0 if is_summary else 1)
        self._apply_chart_tab_style()

    def _apply_chart_tab_style(self):
        for btn in (self.chart_tab_summary, self.chart_tab_analysis):
            btn.setProperty("active", btn.isChecked())
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _render_line_chart(self, days, by_day):
        self.plot.clear()

        self.legend = self.plot.addLegend(offset=(10, 10), labelTextSize="19pt")
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 220))
        self.legend.setPen(pg.mkPen('#e5ecf6'))

        xs = list(range(len(days)))
        xlabels = [d[5:] for d in days]  # MM-DD
        self.plot.getAxis('bottom').setTicks([list(zip(xs, xlabels))])

        exercises = sorted({ex for d in days for ex in by_day[d].keys()})
        if not exercises:
            vb = self.plot.getViewBox()
            vb.setXRange(-0.25, max(0, len(days) - 0.75), padding=0.02)
            vb.setYRange(0, 5, padding=0.12)
            return

        ymax = 0
        for ex in exercises:
            color = self._ex_color.get(ex) or self._assign_color(ex)
            pen = pg.mkPen(color, width=3)
            base = pg.mkColor(color)
            brush = pg.mkBrush(base.red(), base.green(), base.blue(), 55)

            ys = [by_day[d].get(ex, 0) for d in days]
            ymax = max(ymax, max(ys or [0]))

            self.plot.plot(
                xs, ys,
                pen=pen,
                name=self._ex_display(ex),
                antialias=True,
                fillLevel=0,
                brush=brush
            )

            scatter = pg.ScatterPlotItem(
                size=9, brush=pg.mkBrush(color), pen=pg.mkPen('#ffffff', width=2)
            )
            scatter.addPoints([{'pos': (i, y)} for i, y in enumerate(ys)])
            self.plot.addItem(scatter)

        vb = self.plot.getViewBox()
        vb.setXRange(-0.25, len(days) - 0.75, padding=0.02)
        vb.setYRange(0, max(5, ymax) * 1.15 if ymax > 0 else 5, padding=0.12)

    def _fill_path(self, xs, ys):
        from PySide6.QtGui import QPainterPath
        from PySide6.QtCore import QPointF
        if not xs:
            return None
        p = QPainterPath()
        p.moveTo(QPointF(xs[0], 0))
        for x, y in zip(xs, ys):
            p.lineTo(QPointF(x, y))
        p.lineTo(QPointF(xs[-1], 0))
        p.closeSubpath()
        return p

    def _ensure_colors(self, exercises):
        i = 0
        for ex in exercises:
            if ex not in self._ex_color:
                self._ex_color[ex] = self.BASE_COLORS[i % len(self.BASE_COLORS)]
                i += 1

    def _assign_color(self, ex):
        c = self.BASE_COLORS[len(self._ex_color) % len(self.BASE_COLORS)]
        self._ex_color[ex] = c
        return c

    def _ex_display(self, key):
        mapping = {
            "squat": "스쿼트",
            "leg_raise": "레그 레이즈",
            "pushup": "푸시업",
            "shoulder_press": "숄더 프레스",
            "side_lateral_raise": "사레레",
            "bentover_dumbbell": "덤벨 로우",
            "burpee": "버피",
            "jumping_jack": "점핑잭",
        }
        return mapping.get(key, key)

    def _fmt_date(self, dt):
        if not dt:
            return "—"
        try:
            if isinstance(dt, str):
                return dt.split("T")[0]
            return dt.strftime("%Y년 %m월 %d일")
        except Exception:
            return str(dt)

    def _show_profile_only(self):
        try:
            self.plot.clear()
        except Exception:
            pass

    def _show_logged_out_view(self):
        self.lbl_name.setText("—")
        self.lbl_join_date.setText("—")
        self.lbl_until_date.setText("—")
        self.lbl_last_workout.setText("—")
        self.pb_days.setValue(0)
        self.btn_logout.setEnabled(False)
        ret = QMessageBox.question(
            self, "안내", "회원가입 하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self._goto("enroll")
        else:
            self._goto("start")

    def _logout(self):
        self.ctx.clear_current_user()
        QMessageBox.information(self, "로그아웃", "로그아웃되었습니다.")
        self._goto("start")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)
