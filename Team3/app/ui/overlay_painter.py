from typing import Optional
from PySide6.QtCore import Qt, QSize, QRect, Signal
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap

class VideoCanvas(QWidget):
    _anchor_to_cell = {
        "top-left": (0,0), "top-center": (0,1), "top-right": (0,2),
        "center-left": (1,0), "center": (1,1), "center-right": (1,2),
        "bottom-left": (2,0), "bottom-center": (2,1), "bottom-right": (2,2),
    }

    def __init__(self, min_size: QSize | None = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #000000;")

        self._video = QLabel("로딩중...", self)
        self._video.setAlignment(Qt.AlignCenter)
        self._video.setStyleSheet("color:#FFFFFF; font-size:42px; font-weight:600;")
        if min_size:
            self._video.setMinimumSize(min_size)

        self._overlay_root = QWidget(self)
        self._overlay_root.setAttribute(Qt.WA_StyledBackground, False)
        self._overlay_root.setStyleSheet("background: transparent;")

        grid = QGridLayout(self._overlay_root)
        grid.setContentsMargins(12,12,12,12)
        grid.setSpacing(8)
        for i in range(3):
            grid.setColumnStretch(i, 1)
            grid.setRowStretch(i, 1)

        self._cells = [[QWidget(self._overlay_root) for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                v_align = Qt.AlignTop if r == 0 else (Qt.AlignVCenter if r == 1 else Qt.AlignBottom)
                grid.addWidget(self._cells[r][c], r, c, alignment=v_align)

        self._last_qimage: QImage | None = None
        self._img_w = None
        self._img_h = None

        self._fit_mode = "cover"

    def set_fit_mode(self, mode: str):
        self._fit_mode = "cover" if str(mode).lower() == "cover" else "contain"
        self._position_layers()

    def _compute_target_rect(self) -> QRect:
        return QRect(0, 0, self.width(), self.height())

    def _position_layers(self):
        rect = self._compute_target_rect()
        self._video.setGeometry(rect)
        self._overlay_root.setGeometry(rect)
        self._overlay_root.raise_()

        if self._last_qimage is not None and rect.width() > 0 and rect.height() > 0:
            aspect_flag = Qt.KeepAspectRatioByExpanding if self._fit_mode == "cover" else Qt.KeepAspectRatio
            pm = QPixmap.fromImage(self._last_qimage).scaled(
                rect.width(), rect.height(), aspect_flag, Qt.SmoothTransformation
            )
            self._video.setPixmap(pm)
            self._video.setAlignment(Qt.AlignCenter)

    def set_frame(self, qimage: QImage):
        if qimage is None or qimage.isNull():
            self._last_qimage = None
            self._img_w = self._img_h = None
            self._video.clear()
            return
        self._last_qimage = qimage.copy()
        self._img_w = self._last_qimage.width()
        self._img_h = self._last_qimage.height()
        self._position_layers()

    # ------ 오버레이 추가 ------
    def add_overlay(self, widget: QWidget, anchor: str = "top-right"):
        r, c = self._anchor_to_cell.get(anchor, (0, 2))
        cell = self._cells[r][c]
        lay = cell.layout()
        if lay is None:
            lay = QHBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

        if anchor.endswith("left"):
            h_align = Qt.AlignLeft
        elif anchor.endswith("center"):
            h_align = Qt.AlignHCenter
        else:
            h_align = Qt.AlignRight

        lay.addWidget(widget, 0, h_align)
        widget.show()

    def clear_overlays(self):
        for row in self._cells:
            for cell in row:
                lay = cell.layout()
                if lay is not None:
                    while lay.count():
                        item = lay.takeAt(0)
                        w = item.widget()
                        if w is not None:
                            w.hide()
                            w.setParent(self._overlay_root)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._position_layers()

class ExerciseCard(QWidget):
    def __init__(self, title: str = "휴식중", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 80);
                border-radius: 66px;
            }
            QLabel#caption {
                color: rgba(255, 255, 255, 0.9);
                font-size: 40px;
                font-weight: 500;
                letter-spacing: 1.5px;
                background: transparent;
            }
            QLabel#titleValue {
                color: #FFFFFF;
                font-size: 120px;
                font-weight: 600;
                letter-spacing: 2px;
                background: transparent;
            }
            QLabel#countValue {
                color: #00E0FF;
                font-size: 200px;
                font-weight: 600;
                letter-spacing: 3px;
                background: transparent;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(6)

        self._lbl_caption_title = QLabel("운동 종류", self)
        self._lbl_caption_title.setObjectName("caption")
        self._lbl_title_value = QLabel(title, self)
        self._lbl_title_value.setObjectName("titleValue")

        self._lbl_caption_count = QLabel("운동 횟수", self)
        self._lbl_caption_count.setObjectName("caption")
        self._lbl_count_value = QLabel("0", self)
        self._lbl_count_value.setObjectName("countValue")

        lay.addWidget(self._lbl_caption_title)
        lay.addWidget(self._lbl_title_value)
        lay.addSpacing(6)
        lay.addWidget(self._lbl_caption_count)
        lay.addWidget(self._lbl_count_value)

    def set_title(self, title: str):
        self._lbl_title_value.setText(title)

    def set_count(self, n: int):
        self._lbl_count_value.setText(str(int(n)))

class ScoreAdvicePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 80);
                border-radius: 66px;
            }
            QLabel#caption {
                color: rgba(255, 255, 255, 0.9);
                font-size: 40px;
                font-weight: 500;
                letter-spacing: 1.5px;
                background: transparent;
            }
            QLabel#score {
                color: #FFD166;
                font-size: 230px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#advice {
                color: #FFFFFF;
                font-size: 40px;
                font-weight: 500;
                background: transparent;
            }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(8)

        self._lbl_caption = QLabel("평균 점수", self)
        self._lbl_caption.setObjectName("caption")
        self._lbl_score = QLabel("0", self)
        self._lbl_score.setObjectName("score")
        self._lbl_advice = QLabel("", self)
        self._lbl_advice.setObjectName("advice")
        self._lbl_advice.setWordWrap(True)

        lay.addWidget(self._lbl_caption)
        lay.addWidget(self._lbl_score)
        lay.addSpacing(6)
        lay.addWidget(self._lbl_advice)

        self._overlay_container = None

    def set_avg(self, v: float | int):
        try:
            v = int(round(float(v)))
        except Exception:
            v = 0
        self._lbl_score.setText(str(v))

    def set_advice(self, text: str):
        self._lbl_advice.setText(text or "")

class ActionButtons(QWidget):
    endClicked = Signal()
    infoClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setStyleSheet("""
            QWidget { background: transparent; }

            QPushButton {
                border: none;
                border-radius: 14px;
                padding: 14px 22px;
                font-size: 20px;
                font-weight: 600;
            }

            QPushButton#btn-info {
                background: rgba(34, 197, 94, 0.8);  
                color: #FFFFFF;
                border: 2px solid rgba(255,255,255,0.22);
            }
            QPushButton#btn-info:hover {
                background: rgba(255,255,255,0.22);
            }
            QPushButton#btn-info:pressed {
                background: rgba(255,255,255,0.30);
            }

            QPushButton#btn-end {
                background: #FF4D4F;
                color: #FFFFFF; 
            }
            QPushButton#btn-end:hover {
                background: #FF6B6D;
                color: #FFFFFF; 
            }
            QPushButton#btn-end:pressed {
                background: #D9363E;
                color: #FFFFFF; 
            }
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self._btn_info = QPushButton("내 정보", self)
        self._btn_info.setObjectName("btn-info")

        self._btn_end = QPushButton("운동 종료", self)
        self._btn_end.setObjectName("btn-end")

        lay.addWidget(self._btn_info)
        lay.addWidget(self._btn_end)

        self._btn_end.clicked.connect(self.endClicked.emit)
        self._btn_info.clicked.connect(self.infoClicked.emit)

        self._btn_end.setMinimumHeight(100)
        self._btn_info.setMinimumHeight(100)

    def set_enabled(self, end_enabled: bool = True, info_enabled: bool = True):
        self._btn_end.setEnabled(end_enabled)
        self._btn_info.setEnabled(info_enabled)

class PoseAnglePanel(QWidget):
    ROW_KEYS = [
        ("Knee",      "Knee(L)",      "Knee(R)"),
        ("Hip",       "Hip(L)",       "Hip(R)"),
        ("Shoulder",  "Shoulder(L)",  "Shoulder(R)"),
        ("Elbow",     "Elbow(L)",     "Elbow(R)"),
        ("HipLine",   "HipLine(L)",   "HipLine(R)"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 160);
                border-radius: 16px;
            }
            QLabel#hdr {
                color: #BFC6CF;
                font-size: 16px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#cell {
                color: #FFFFFF;
                font-size: 16px;
                font-weight: 600;
                background: transparent;
            }
            QLabel#cell-dim {
                color: #BFC6CF;
                font-size: 14px;
                font-weight: 600;
                background: transparent;
            }
            QLabel#title {
                color: #FFFFFF;
                font-size: 18px;
                font-weight: 800;
                letter-spacing: 0.5px;
                background: transparent;
            }
        """)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(14, 12, 14, 12)
        wrap.setSpacing(8)

        title = QLabel("Pose Angles (°)", self)
        title.setObjectName("title")
        wrap.addWidget(title)

        grid_host = QWidget(self)
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # headers
        hdrs = ["Part", "L", "R", "Avg"]
        for c, h in enumerate(hdrs):
            lab = QLabel(h, self)
            lab.setObjectName("hdr")
            grid.addWidget(lab, 0, c)

        # cells
        self._cells: dict[str, QLabel] = {}
        for r, (part, lkey, rkey) in enumerate(self.ROW_KEYS, start=1):
            # part name
            lab_part = QLabel(part, self)
            lab_part.setObjectName("cell-dim")
            grid.addWidget(lab_part, r, 0)

            # L / R / Avg cells
            for c, col in enumerate(("L", "R", "Avg"), start=1):
                lab = QLabel("-", self)
                lab.setObjectName("cell")
                grid.addWidget(lab, r, c)
                self._cells[f"{part}:{col}"] = lab

        wrap.addWidget(grid_host)

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "-"
        try:
            return f"{float(v):.1f}"
        except Exception:
            return "-"

    def set_angles(self, angles: dict):
        for part, lkey, rkey in self.ROW_KEYS:
            l = angles.get(lkey)
            r = angles.get(rkey)
            avg = None
            if isinstance(l, (int, float)) and isinstance(r, (int, float)):
                avg = (float(l) + float(r)) / 2.0

            self._cells[f"{part}:L"].setText(self._fmt(l))
            self._cells[f"{part}:R"].setText(self._fmt(r))
            self._cells[f"{part}:Avg"].setText(self._fmt(avg))

class AIMetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aimetrics")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setStyleSheet("""
            #aimetrics {
                background: rgba(0, 0, 0, 80);   
                border-radius: 16px;
            }
            #aimetrics QWidget {
                background: transparent;
            }
            QLabel#title {
                color: #FFFFFF; font-size: 50px; font-weight: 500; letter-spacing: 0.5px;
                background: transparent;
            }
            QLabel#dim {
                color: #BFC6CF; font-size: 25px; font-weight: 500; background: transparent;
            }
            QLabel#val {
                color: #FFFFFF; font-size: 25px; font-weight: 500; background: transparent;
            }
        """)

        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(14, 12, 14, 12)
        wrap.setSpacing(8)

        title = QLabel("근전도 센서 데이터", self)
        title.setObjectName("title")
        wrap.addWidget(title)

        grid_host = QWidget(self)
        grid_host.setObjectName("grid_host")
        grid_host.setAttribute(Qt.WA_StyledBackground, False)   

        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(6)

        rows = [
            ("템포점수",     "tempo_score"),
            ("템포판정",     "tempo_level"),
            ("상태",        "imu_state"),
            ("L_피로도",      "fi_l"),
            ("R_피로도",      "fi_r"),
            ("L_피로도판정",  "stage_l"),
            ("R_피로도판정",  "stage_r"),
            ("불균형",        "bi"),
            ("불균형판정",    "bi_stage"),
            ("불균형설명",    "bi_text"),
        ]

        self._cells = {}
        for r, (lab, key) in enumerate(rows):
            k = QLabel(lab, self); k.setObjectName("dim")
            v = QLabel("-",  self); v.setObjectName("val")
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
            self._cells[key] = v

        wrap.addWidget(grid_host)

    @staticmethod
    def _fmt_num(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "-"

    def set_imu(self, tempo_score=None, tempo_level=None, imu_state=None):
        self._cells["tempo_score"].setText("-" if tempo_score is None else str(int(tempo_score)))
        self._cells["tempo_level"].setText("-" if tempo_level is None else str(tempo_level))
        self._cells["imu_state"].setText("-" if imu_state is None else str(imu_state))

    def set_ai(self, fi_l=None, fi_r=None, stage_l=None, stage_r=None,
            bi=None, bi_stage=None, bi_text=None):
        self._cells["fi_l"].setText(self._fmt_num(fi_l) if fi_l is not None else "-")
        self._cells["fi_r"].setText(self._fmt_num(fi_r) if fi_r is not None else "-")
        self._cells["stage_l"].setText(str(stage_l) if stage_l is not None else "-")
        self._cells["stage_r"].setText(str(stage_r) if stage_r is not None else "-")
        self._cells["bi"].setText(self._fmt_num(bi) if bi is not None else "-")
        self._cells["bi_stage"].setText(str(bi_stage) if bi_stage is not None else "-")
        self._cells["bi_text"].setText(str(bi_text) if bi_text is not None else "-")

