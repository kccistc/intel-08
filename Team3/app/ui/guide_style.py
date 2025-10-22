from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QScrollArea, QFrame, QLabel

COLOR_BG_PAGE   = "#ffffff"
COLOR_BG_PANEL  = "#f3f9ff"
COLOR_BORDER    = "#e5ecf6"
COLOR_PRIMARY_L = "#eef6ff"
COLOR_PRIMARY   = "#d6e8ff"
COLOR_PRIMARY_D = "#24527a"
COLOR_TEXT_900  = "#0f172a"
COLOR_TEXT_700  = "#374151"
COLOR_TEXT_600  = "#6b7380"
COLOR_ACCENT    = "#7aa2ff"
COLOR_HOVER     = "#bcd4ff"
COLOR_SB_HANDLE = "#cfe2ff"

def force_bg(widget: QWidget, css: str) -> None:
    widget.setAttribute(Qt.WA_StyledBackground, True)
    widget.setStyleSheet(css)

def style_page_root(page: QWidget) -> None:
    force_bg(page, f"background:{COLOR_BG_PAGE};")

# 왼쪽 패널
def style_side_panel(panel: QFrame) -> None:
    panel.setObjectName("SidePanel")
    force_bg(panel, f"QFrame#SidePanel {{ background:{COLOR_BG_PANEL}; border-right:1px solid #eef2f7; }}")

# 스크롤 영역
def style_scrollarea(scroll: QScrollArea) -> None:
    scroll.setObjectName("GuideListScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFocusPolicy(Qt.NoFocus)

    scroll.setStyleSheet(f"""
        QScrollArea#GuideListScroll {{
            background:{COLOR_BG_PANEL};
            border:none;
        }}
        QScrollArea#GuideListScroll > QWidget {{
            background:{COLOR_BG_PANEL};
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 8px 4px 8px 0;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {COLOR_SB_HANDLE};
            border-radius: 4px;
            min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {COLOR_HOVER}; }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0px; background: transparent; border: none;
        }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
    """)

# 운동 리스트
def style_exercise_card(card: QFrame) -> None:
    card.setObjectName("ExerciseCard")
    force_bg(card, f"""
        QFrame#ExerciseCard {{
            background:#ffffff; border:1px solid {COLOR_BORDER}; border-radius:14px;
        }}
        QFrame#ExerciseCard[selected="true"] {{ border:2px solid {COLOR_ACCENT}; }}
        QFrame#ExerciseCard:hover {{ border-color:{COLOR_HOVER}; }}
        QLabel#title {{ font-size:20px; font-weight:700; color:#1f2937; background:transparent; }}
        QLabel#sets  {{ color:{COLOR_TEXT_600}; background:transparent; }}
        QLabel#chip  {{
            background:{COLOR_PRIMARY_L}; border:1px solid {COLOR_PRIMARY}; border-radius:999px;
            padding:2px 8px; color:{COLOR_PRIMARY_D}; font-size:16px; font-weight:600;
        }}
    """)

# 정보 패널
def style_info_card(card: QFrame) -> None:
    card.setObjectName("InfoCard")
    force_bg(card, f"""
        QFrame#InfoCard {{
            background:#ffffff;
            border:1px solid {COLOR_BORDER};
            border-radius:16px;
        }}
    """)

# 헤더 라벨
def style_header_title(lbl: QLabel) -> None:
    lbl.setStyleSheet(f"font-size:80px; font-weight:900; color:{COLOR_TEXT_900}; background:transparent;")

def style_header_chip(lbl: QLabel) -> None:
    lbl.setObjectName("HeaderChip")
    lbl.setStyleSheet(
      f"background:{COLOR_PRIMARY_L}; border:1px solid {COLOR_PRIMARY}; border-radius:999px;"
      "padding:6px 12px;"
      f"color:{COLOR_PRIMARY_D}; font-size:22px; font-weight:700;"
    )

def style_header_desc(lbl: QLabel) -> None:
    lbl.setStyleSheet(f"color:{COLOR_TEXT_700}; font-size:20px; margin:0; background:transparent;")

def style_muted_text(lbl: QLabel) -> None:
    lbl.setStyleSheet(f"background:transparent; color:{COLOR_TEXT_700};")
