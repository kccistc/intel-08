from typing import List, Iterable
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QSpacerItem
)
from core.page_base import PageBase
from data.guide_data import Exercise, list_all
from ui.guide_style import (
    style_page_root, style_side_panel, style_scrollarea, style_exercise_card,
    style_header_title, force_bg
)
import os

# ---------------- UI Util (배경 자원 경로) ----------------
def asset_path(*parts) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, *parts)

def _clear_layout(layout) -> None:
    while layout and layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        c = item.layout()
        if w is not None:
            w.deleteLater()
        elif c is not None:
            _clear_layout(c)

class ExerciseCard(QFrame):
    def __init__(self, info: Exercise, parent: QWidget | None = None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("ExerciseCard")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(6)

        title_line = QHBoxLayout()
        title_line.setContentsMargins(0, 0, 0, 0)
        title_line.setSpacing(8)

        title = QLabel(info.title)
        title.setObjectName("title")
        title.setStyleSheet("background:transparent;")
        title.setWordWrap(False)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        cat = QLabel(info.category)
        cat.setObjectName("chip")
        cat.setStyleSheet("background:transparent;")

        title_line.addWidget(title, 1)
        title_line.addWidget(cat, 0, Qt.AlignVCenter)
        root.addLayout(title_line)

        # 칩 스타일 포함
        force_bg(self, """
            QFrame#ExerciseCard {
                background:#ffffff;
                border:1px solid rgba(0,0,0,0.06);
                border-radius:16px;
                min-height:100px;
                padding:18px 24px;
            }
            QFrame#ExerciseCard[selected="true"] {
                background:#1a73e8;
                border:1px solid #1a73e8;
            }
            QFrame#ExerciseCard[selected="true"] QLabel#title,
            QFrame#ExerciseCard[selected="true"] QLabel#chip { color:white; }

            QLabel#title { font-size:28px; font-weight:700; color:#0f172a; }
            QLabel#chip  {
                background:#eef4ff;
                border:1px solid rgba(25,118,210,0.5);
                border-radius:999px;
                padding:2px 10px;
                color:#24527a;
                font-size:14px;
                font-weight:600;
            }
        """)

    def setSelected(self, v: bool) -> None:
        self.setProperty("selected", v)
        self.style().unpolish(self)
        self.style().polish(self)

def bullet_list(items: Iterable[str], numbered: bool = False) -> QWidget:
    w = QWidget()
    force_bg(w, "background:transparent;")
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    for i, t in enumerate(items, 1):
        line = QHBoxLayout()
        line.setSpacing(10)
        dot = QLabel(str(i) if numbered else "•")
        dot.setStyleSheet("background:transparent; color:#4b5563; font-size:20px;")
        dot.setFixedWidth(24)
        lbl = QLabel(t)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("""
            background:rgba(25,118,210,0.10);
            color:#0f172a; font-size:25px; font-weight:500; border-radius:12px; padding:10px 12px;
        """)
        line.addWidget(dot)
        line.addWidget(lbl, 1)
        v.addLayout(line)
    return w

class GuidePage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("GuidePage")

        # ===== 배경 =====
        self._bg_pix = None
        self._bg = QLabel(self)
        self._bg.setScaledContents(False)
        self._bg.lower()

        bg_path = asset_path("assets", "background", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pm = QPixmap(bg_path)
            if not pm.isNull():
                self._bg_pix = pm
                self._rescale_bg()

        # 글래스 패널
        self._panel = QFrame(self)
        self._panel.setObjectName("glassPanel")
        self._panel.setAttribute(Qt.WA_StyledBackground, True)

        style_page_root(self)

        self.exercises = list_all()

        # === 패널 내부 루트 레이아웃 ===
        panel_root = QVBoxLayout(self._panel)
        panel_root.setContentsMargins(28, 28, 28, 28)
        panel_root.setSpacing(18)

        # ---------- TopBar ----------
        top = QFrame()
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(22, 18, 22, 18)
        top_lay.setSpacing(14)

        self.btn_back = QPushButton("←")
        self.btn_back.setObjectName("BtnBack")
        self.btn_back.setFixedHeight(56)
        self.btn_back.clicked.connect(lambda: self._goto("start"))

        self.title = QLabel("운동 가이드")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        top_lay.addWidget(self.btn_back)
        top_lay.addSpacing(12)
        top_lay.addWidget(self.title, 1)

        self.btn_user = QPushButton("로그인 필요")
        self.btn_user.setObjectName("BtnUser")
        self.btn_user.setFixedHeight(48)
        self.btn_user.clicked.connect(lambda: self._goto("info"))

        self.btn_top_start = QPushButton("▶  운동 시작")
        self.btn_top_start.setObjectName("BtnStart")
        self.btn_top_start.setFixedHeight(48)
        self.btn_top_start.clicked.connect(lambda: self._goto("exercise"))

        top_lay.addWidget(self.btn_user)
        top_lay.addWidget(self.btn_top_start)

        # ---------- 본문 ----------
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(18)

        # 좌측 패널 (리스트)
        self.left_panel = self._build_left_panel()
        body.addWidget(self.left_panel, 2)

        # 우측 디테일 패널 (운동 제목 + 방법/주의사항)
        self.detail_panel = self._build_detail_panel()
        body.addWidget(self.detail_panel, 3)

        panel_root.addWidget(top)
        panel_root.addLayout(body, 1)

        # 스타일시트 적용
        self.setStyleSheet(self._stylesheet())

        if self.exercises:
            self._select(self.exercises[0])

    def on_enter(self, ctx):
        self._set_user_button(ctx)

    # ---------- 좌측 패널 ----------
    def _build_left_panel(self) -> QWidget:
        side = QFrame()
        side.setObjectName("LeftMenu")
        style_side_panel(side)

        side.setFixedWidth(440)

        v = QVBoxLayout(side)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(12)

        scroll = QScrollArea()
        style_scrollarea(scroll)

        content = QWidget()
        force_bg(content, "background:transparent;")
        lv = QVBoxLayout(content)
        lv.setContentsMargins(8, 8, 4, 8)
        lv.setSpacing(12)

        self._cards: List[ExerciseCard] = []
        for ex in self.exercises:
            card = ExerciseCard(ex)
            style_exercise_card(card)
            card.mousePressEvent = lambda e, _ex=ex: self._select(_ex)
            self._cards.append(card)
            lv.addWidget(card)

        lv.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        v.addWidget(scroll, 1)
        return side

    # ---------- 우측 패널 ----------
    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        force_bg(panel, "background:transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # 제목 (카테고리/설명 없음)
        self.h_title = QLabel("")
        self.h_title.setObjectName("HeaderTitle")
        style_header_title(self.h_title)

        # ------- 운동 방법 카드 (steps) -------
        self.steps_card = QFrame()
        self.steps_card.setObjectName("StepsCard")
        steps_box = QVBoxLayout(self.steps_card)
        steps_box.setContentsMargins(22, 22, 22, 22)
        steps_box.setSpacing(12)

        steps_title = QLabel("운동 방법")
        steps_title.setObjectName("StepsTitle")
        self.steps_widget = bullet_list([], numbered=True)

        steps_box.addWidget(steps_title)
        steps_box.addWidget(self.steps_widget)

        # ------- 주의사항 카드 (tips) -------
        self.tips_card = QFrame()
        self.tips_card.setObjectName("TipsCard")
        tips_box = QVBoxLayout(self.tips_card)
        tips_box.setContentsMargins(22, 22, 22, 22)
        tips_box.setSpacing(12)

        tips_title = QLabel("주의사항 및 팁")
        tips_title.setObjectName("TipsTitle")
        self.tips_widget = bullet_list([], numbered=False)

        tips_box.addWidget(tips_title)
        tips_box.addWidget(self.tips_widget)

        layout.addWidget(self.h_title)
        layout.addWidget(self.steps_card, 1)
        layout.addWidget(self.tips_card, 1)

        return panel

    def _select(self, ex: Exercise) -> None:
        for c in self._cards:
            c.setSelected(c.info.key == ex.key)
        self.h_title.setText(ex.title)
        self._replace_bullet(self.steps_widget, ex.steps, True)
        self._replace_bullet(self.tips_widget, ex.tips, False)

    def _replace_bullet(self, container: QWidget, items: Iterable[str], numbered: bool = False) -> None:
        lay = container.layout()
        _clear_layout(lay)
        for i, t in enumerate(items, 1):
            row = QHBoxLayout()
            row.setSpacing(10)
            dot = QLabel(str(i) if numbered else "•")
            dot.setFixedWidth(24)
            dot.setStyleSheet("background:transparent; color:#4b5563; font-size:20px;")
            lbl = QLabel(t)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("""
                background:rgba(25,118,210,0.10); color:#0f172a; font-size:25px; font-weight:500;
                border-radius:12px; padding:10px 12px;
            """)
            row.addWidget(dot)
            row.addWidget(lbl, 1)
            lay.addLayout(row)

    def _goto(self, page: str) -> None:
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    # ===== 배경/패널 레이아웃 처리 =====
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
        self._panel.setGeometry(x, y, target_w, target_h)

    def _rescale_bg(self):
        if self._bg_pix:
            self._bg.setGeometry(self.rect())
            scaled = self._bg_pix.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self._bg.setPixmap(scaled)

    def _set_user_button(self, ctx) -> None:
        try:
            is_in = getattr(ctx, "is_logged_in", None)
            logged_in = bool(is_in() if callable(is_in) else getattr(ctx, "current_user_id", None))
            if logged_in:
                name = getattr(ctx, "current_user_name", None) or "사용자"
                self.btn_user.setText(f"{name} 님")
            else:
                self.btn_user.setText("로그인")
        except Exception:
            self.btn_user.setText("로그인")

    # ===== 스타일 =====
    def _stylesheet(self) -> str:
        return """
        #glassPanel {
            background: rgba(255,255,255,1);
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.25);
        }

        #TopBar {
            background: #1976d2;
            border-radius: 20px;
        }
        #Title {
            color: white;
            font-size: 44px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        #BtnBack {
            background: rgba(255,255,255,0.18);
            color: white;
            border: none;
            padding: 0 22px;
            border-radius: 14px;
            font-size: 28px;
            font-weight: 600;
        }
        #BtnBack:hover { background: rgba(255,255,255,0.28); }
        #BtnUser {
            background: rgba(255,255,255,0.20);
            color: white;
            border: none;
            padding: 0 22px;
            border-radius: 14px;
            font-size: 18px;
            font-weight: 600;
        }
        #BtnUser:hover { background: rgba(255,255,255,0.28); }
        #BtnStart {
            background: #17c964;
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 14px;
            font-size: 18px;
            font-weight: 700;
        }
        #BtnStart:hover { background: #11b85a; }

        #LeftMenu {
            background: rgba(255,255,255,0.90);
            border-radius: 18px;
            min-width: 420px;
            max-width: 420px;
        }

        #StepsCard, #TipsCard {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 18px;
        }
        #StepsTitle, #TipsTitle {
            color: #111827;
            font-size: 28px;
            font-weight: 700;
        }
        """
