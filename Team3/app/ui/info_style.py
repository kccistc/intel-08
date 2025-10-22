from PySide6.QtWidgets import QWidget

# 색/타이포 일괄 정의
TOKENS = {
    "bg_app":      "#f5f8ff",
    "fg_text":     "#465065",
    "fg_title":    "#0f172a",
    "fg_subtitle": "#1f2937",
    "fg_muted":    "#6b7380",
    "fg_chip":     "#24527a",
    "fg_btn":      "#334155",
    "fg_kpi":      "#111827",

    "card_border": "#e8eef8",
    "group_border":"#e5ecf6",

    "chip_bg":     "#eef6ff",
    "chip_border": "#d6e8ff",

    "btn_grad_top":"#ffffff",
    "btn_grad_bot":"#eef4ff",
    "btn_border":  "#d7e3f6",

    "accent":      "#7c8cf8",
    "accent_sub":  "#1e40af",

    "bar_bg":      "#e9eef7",
}

# 공통 QSS (InfoPage 전체 + Card 등)
BASE_QSS = f"""
/* 페이지 루트 */
QWidget#InfoPage {{
    background: {TOKENS["bg_app"]};
}}

/* 라벨 기본: 25px / 500 */
QLabel {{
    color: {TOKENS["fg_text"]};
    font-size: 25px;
    font-weight: 500;
}}

/* 제목류(굵기 700) */
QLabel[cls="headline"] {{
    color: {TOKENS["fg_title"]};
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 0.4px;
}}
QLabel[cls="title"] {{
    color: {TOKENS["fg_subtitle"]};
    font-size: 25px;
    font-weight: 700;
}}

/* 칩/서브톤 */
QLabel[cls="muted"] {{
    color: {TOKENS["fg_muted"]};
    font-size: 25px;
    font-weight: 500;
}}
QLabel[cls="chip"] {{
    background: {TOKENS["chip_bg"]};
    color: {TOKENS["fg_chip"]};
    border: 1px solid {TOKENS["chip_border"]};
    padding: 8px 12px;
    border-radius: 10px;
    font-weight: 500;
    font-size: 25px;
}}

/* KPI 큰 숫자 */
QLabel[cls="kpi"] {{
    color: {TOKENS["fg_kpi"]};
    font-size: 26px;
    font-weight: 700;
}}

/* 아이콘 라벨 */
QLabel[cls="icon"] {{
    color: {TOKENS["accent"]};
    font-size: 25px;
    font-weight: 700;
}}

/* 그룹박스(필요 시) */
QGroupBox {{
    background: #fff;
    border: 1px solid {TOKENS["group_border"]};
    border-radius: 16px;
    margin-top: 16px;
    color: #3b3f53;
    font-weight: 700;
    font-size: 25px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 6px 10px;
    color: #556074;
    background: rgba(138,180,248,0.20);
    border-radius: 10px;
    font-weight: 700;
    font-size: 25px;
}}

/* 버튼(25px / 500) */
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {TOKENS["btn_grad_top"]}, stop:1 {TOKENS["btn_grad_bot"]});
    color: {TOKENS["fg_btn"]};
    border: 1px solid {TOKENS["btn_border"]};
    border-radius: 12px;
    padding: 12px 22px;
    font-weight: 500;
    font-size: 25px;
}}
QPushButton:hover {{ background:#f3f7ff; }}

/* 카드 공통 */
QFrame#Card {{
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid {TOKENS["card_border"]};
}}

/* 진행바 색 */
QProgressBar {{
    background: {TOKENS["bar_bg"]};
    border-radius: 6px;
    height: 12px;
}}
QProgressBar::chunk {{
    background: {TOKENS["accent"]};
    border-radius: 6px;
}}
"""

def apply_info_page_styles(root: QWidget) -> None:
    """
    InfoPage 전용 스타일 일괄 적용.
    """
    if root.objectName() != "InfoPage":
        root.setObjectName("InfoPage")
    root.setStyleSheet(BASE_QSS)

ADDITIONAL = f"""
QLabel[cls="display"] {{
    color: {TOKENS["fg_title"]};
    font-size: 44px;   
    font-weight: 700;
}}

QLabel[cls="date"] {{
    color: {TOKENS["fg_kpi"]};
    font-size: 32px;   
    font-weight: 700;
}}
"""

def apply_info_page_styles(root: QWidget) -> None:
    if root.objectName() != "InfoPage":
        root.setObjectName("InfoPage")
    root.setStyleSheet(BASE_QSS + ADDITIONAL)