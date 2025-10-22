import os
import cv2
import numpy as np

from core.page_base import PageBase
from ui.virtual_keyboard_ko import VirtualKeyboardKO  
from PySide6.QtCore import Qt, QSize, QRect, QEvent, QPoint, QTimer, Signal
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QGraphicsDropShadowEffect, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem
)

def asset_path(*parts) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, *parts)

def first_existing(path_without_ext: str):
    for ext in (".png", ".svg", ".jpg", ".jpeg", ""):
        p = path_without_ext + ext
        if os.path.exists(p):
            return p
    return None

class EnrollPage(PageBase): 
    finished = Signal()       

    def __init__(self, parent=None, target_n: int = 20):
        super().__init__(parent)
        self.ctx = None
        self.stage = 1  
        self._bg_pix = None
        self._current_name = ""

        self.target_n = target_n
        self.collecting = False
        self.collected = []
        self._last_qimg = None

        # 주기적 카메라 렌더/수집
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # ---- UI 구성 ----
        self._build_ui()
        self._try_load_assets()
        self._render_stage()  # 초기 렌더

    # ---------------- UI Build ----------------
    def _build_ui(self):
        self.setObjectName("EnrollPage")

        self.bg = QLabel(self)
        self.bg.setScaledContents(False)
        self.bg.lower()

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(0)
        root.addStretch(1)

        self.panel = self._make_glass_panel()
        root.addWidget(self.panel, alignment=Qt.AlignCenter)
        root.addStretch(1)

        self.setStyleSheet(self._stylesheet())

        self.vkbd = VirtualKeyboardKO(None, parent=self)
        self.vkbd.hide()
        self.vkbd.hiddenRequested.connect(lambda: self.nameEdit.clearFocus())
        self.vkbd.finished.connect(lambda: self.nameEdit.clearFocus())

        self.nameEdit.installEventFilter(self)
        self.installEventFilter(self)

    def _make_glass_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("glassPanel")
        panel.setAttribute(Qt.WA_StyledBackground, True)

        shadow = QGraphicsDropShadowEffect(panel)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 16)
        shadow.setColor(Qt.black)
        panel.setGraphicsEffect(shadow)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(56, 44, 56, 44)

        topbar = QHBoxLayout()
        topbar.setSpacing(0)
        self.backBtn = QPushButton()
        self.backBtn.setObjectName("backBtn")
        self.backBtn.setFixedSize(90, 90)
        self.backBtn.setCursor(Qt.PointingHandCursor)
        self.backBtn.setToolTip("뒤로 가기")
        self.backBtn.clicked.connect(self._on_back)
        topbar.addWidget(self.backBtn, alignment=Qt.AlignLeft)
        topbar.addStretch(1)
        outer.addLayout(topbar)

        # 아이콘/타이틀
        self.icon_img = QLabel(panel)
        self.icon_img.setObjectName("iconImg")
        self.icon_img.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.icon_img, alignment=Qt.AlignHCenter)

        self.title = QLabel("회원가입", panel)
        self.title.setObjectName("titleLabel")
        self.title.setAlignment(Qt.AlignHCenter)
        outer.addWidget(self.title)

        step_row = QHBoxLayout()
        step_row.setSpacing(12)
        self.step1 = QLabel("1. 정보 입력")
        self.step1.setObjectName("stepPillActive")
        self.step1.setAlignment(Qt.AlignCenter)
        self.step1.setFixedHeight(34)
        self.step1.setMinimumWidth(120)

        dash = QLabel("—")
        dash.setObjectName("dash")
        dash.setAlignment(Qt.AlignCenter)

        self.step2 = QLabel("2. 얼굴 인식")
        self.step2.setObjectName("stepPill")
        self.step2.setAlignment(Qt.AlignCenter)
        self.step2.setFixedHeight(34)
        self.step2.setMinimumWidth(120)

        step_row.addStretch(1)
        step_row.addWidget(self.step1)
        step_row.addWidget(dash)
        step_row.addWidget(self.step2)
        step_row.addStretch(1)
        outer.addLayout(step_row)

        self.hint = QLabel("", panel)
        self.hint.setObjectName("hintLabel")
        self.hint.setAlignment(Qt.AlignHCenter)
        outer.addWidget(self.hint)

        # ---------- Stage 1: 폼 ----------
        self.form_wrap = QWidget(panel)
        self.form_wrap.setObjectName("formWrap")
        self.form_wrap.setMaximumWidth(900)
        form_layout = QVBoxLayout(self.form_wrap)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(10)

        self.fieldLabel = QLabel("이름", self.form_wrap)
        self.fieldLabel.setObjectName("fieldLabel")

        self.nameEdit = QLineEdit(self.form_wrap)
        self.nameEdit.setObjectName("nameEdit")
        self.nameEdit.setPlaceholderText("이름을 입력하세요")
        self.nameEdit.setMinimumHeight(58)
        self.nameEdit.returnPressed.connect(lambda: self.nextBtn.click())

        form_layout.addWidget(self.fieldLabel)
        form_layout.addWidget(self.nameEdit)
        outer.addWidget(self.form_wrap, alignment=Qt.AlignHCenter)

        # ---------- Stage 2: 카메라 영역 ----------
        self.cameraFrame = QFrame(panel)
        self.cameraFrame.setObjectName("cameraFrame")
        self.cameraFrame.setStyleSheet(
            "background: rgba(0,0,0,40);"
            "border: 2px dashed rgba(255,255,255,80);"
            "border-radius: 16px;"
        )
        self.cameraFrame.setMinimumSize(600, 360)
        self.cameraFrame.hide() 

        # 실제 영상 렌더용 라벨(프레임 안에 꽉 차게)
        self.cameraLabel = QLabel(self.cameraFrame)
        self.cameraLabel.setAlignment(Qt.AlignCenter)
        self.cameraLabel.setGeometry(self.cameraFrame.rect())
        self.cameraLabel.setScaledContents(False)

        outer.addWidget(self.cameraFrame, alignment=Qt.AlignHCenter)
        outer.addItem(QSpacerItem(0, 12, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.nextBtn = QPushButton("")
        self.nextBtn.setObjectName("nextBtn")
        self.nextBtn.setFixedHeight(64)
        self.nextBtn.setMaximumWidth(620)
        self.nextBtn.setCursor(Qt.PointingHandCursor)
        self.nextBtn.clicked.connect(self._on_next)
        outer.addWidget(self.nextBtn, alignment=Qt.AlignHCenter)

        return panel

    # ---------------- Stage 전환/표시 ----------------
    def _render_stage(self):
        if self.stage == 1:
            self.form_wrap.show()
            self.cameraFrame.hide()
            self.hint.setText("회원님의 이름을 입력해주세요")
            self.nextBtn.setText("다음 단계")
            self.step1.setObjectName("stepPillActive")
            self.step2.setObjectName("stepPill")
            self._end_collection(stop_timer=False)  # 타이머는 on_leave에서 멈춤
        else:
            self.form_wrap.hide()
            self.cameraFrame.show()
            self.hint.setText("화면의 영역 안에 얼굴을 맞춰주세요")
            self.nextBtn.setText("얼굴 인식 시작")
            self.step1.setObjectName("stepPill")
            self.step2.setObjectName("stepPillActive")
            self.vkbd.hide()
            self.nameEdit.clearFocus()

        self.setStyleSheet(self._stylesheet())

    def _on_next(self):
        if self.stage == 1:
            name = (self.nameEdit.text() or "").strip()
            if not name:
                self.nameEdit.setPlaceholderText("이름을 입력해주세요")
                return
            self._current_name = name
            self.stage = 2
            self._render_stage()
        else:
            if not self.collecting:
                self.collecting = True
                self.collected.clear()
                self.hint.setText("정면을 바라보고 자연스럽게 움직여 주세요 (수집 0/{})".format(self.target_n))
                self.nextBtn.setText("수집 중...")
                self.nextBtn.setEnabled(False)

    def _on_back(self):
        if self.stage == 2:
            if self.collecting:
                self._end_collection()
            self.stage = 1
            self._render_stage()
        else:
            if self.ctx and hasattr(self.ctx, "router"):
                try:
                    self.ctx.router.navigate("start")
                except Exception:
                    pass

    # ---------------- Events ----------------
    def eventFilter(self, obj, ev):
        if obj is self.nameEdit and ev.type() in (QEvent.FocusIn, QEvent.MouseButtonPress):
            if self.stage == 1:
                self._show_keyboard_for(self.nameEdit)

        if obj is self and ev.type() == QEvent.MouseButtonPress and self.vkbd.isVisible():
            pos = ev.position().toPoint()  
            ne_top_left = self.nameEdit.mapTo(self, QPoint(0, 0))
            ne_rect_page = QRect(ne_top_left, self.nameEdit.size())
            if not (self.vkbd.geometry().contains(pos) or ne_rect_page.contains(pos)):
                self.vkbd.hide()
        return super().eventFilter(obj, ev)

    def _show_keyboard_for(self, line_edit: QLineEdit):
        self.vkbd.setTarget(line_edit)
        kb_h = max(220, int(self.height() * 0.3))
        kb_w = int(self.width() * 0.7)
        kb_x = int((self.width() - kb_w) / 2)
        self.vkbd.setGeometry(kb_x, self.height() - kb_h, kb_w, kb_h)
        self.vkbd.show()
        self.vkbd.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_bg()
        self._size_panel()

        if hasattr(self, "nameEdit") and self.panel.width() > 0:
            self.nameEdit.setFixedWidth(int(self.panel.width() * 0.3))

        if hasattr(self, "cameraFrame"):
            pw = self.panel.width()
            ph = self.panel.height()
            cam_w = max(640, int(pw * 0.60))
            cam_h = max(360, int(ph * 0.40))
            self.cameraFrame.setFixedSize(cam_w, cam_h)
            self.cameraLabel.setGeometry(self.cameraFrame.rect())

        if hasattr(self, "vkbd") and self.vkbd.isVisible():
            self._show_keyboard_for(self.nameEdit)

    # ---------------- Assets ----------------
    def _try_load_assets(self):
        # 배경
        bg_path = asset_path("assets", "background", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pm = QPixmap(bg_path)
            if not pm.isNull():
                self._bg_pix = pm
                self._rescale_bg()

        # 아이콘
        icon_path = asset_path("assets", "icon_register.png")
        if os.path.exists(icon_path):
            pm = QPixmap(icon_path)
            if not pm.isNull():
                scaled = pm.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.icon_img.setPixmap(scaled)

        back_path = asset_path("assets", "buttons", "back_arrow.png")
        if back_path:
            self.backBtn.setIcon(QIcon(back_path))
            self.backBtn.setIconSize(QSize(100, 100))

    def _rescale_bg(self):
        if self._bg_pix:
            self.bg.setGeometry(self.rect())
            scaled = self._bg_pix.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self.bg.setPixmap(scaled)

    def _size_panel(self):
        target_w = int(self.width() * 0.90)
        target_h = int(self.height() * 0.90)
        target_w = max(920, target_w)
        target_h = max(620, target_h)
        self.panel.setFixedSize(target_w, target_h)

    # ---------------- Styles ----------------
    def _stylesheet(self) -> str:
        return """
        #EnrollPage { background: transparent; }

        /* glass */
        #glassPanel {
            background: qlineargradient(x1:0 y1:0, x2:1 y2:1,
                stop:0 rgba(0,0,0,160),
                stop:1 rgba(255,255,255,28));
            border-radius: 32px;
            border: 1px solid rgba(255,255,255,40);
        }

        /* back button (아이콘형, hover/pressed만 배경) */
        QPushButton#backBtn {
            border: none;
            background: rgba(255,255,255,0);
            border-radius: 50px;
        }
        QPushButton#backBtn:hover { background: rgba(255,255,255,48); border-radius: 50px; }
        QPushButton#backBtn:pressed { background: rgba(255,255,255,64); border-radius: 50px; }

        /* title */
        #titleLabel {
            color: white;
            font-size: 80px;
            font-weight: 1000;
            letter-spacing: 2px;
        }

        /* steps */
        QLabel#dash { color: rgba(255,255,255,200); font-size: 18px; }
        #stepPill, #stepPillActive {
            padding: 4px 16px;
            border-radius: 18px;
            font-size: 16px;
        }
        #stepPill {
            color: rgba(255,255,255,210);
            background: rgba(255,255,255,42);
        }
        #stepPillActive {
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4ea6ff, stop:1 #66f0c8);
            font-weight: 700;
        }

        /* hint */
        #hintLabel {
            margin-top: 6px;
            color: rgba(255,255,255,220);
            font-size: 18px;
        }

        /* form label */
        #fieldLabel {
            color: rgba(255,255,255,230);
            font-size: 18px;
            font-weight: 700;
        }

        /* line edit */
        QLineEdit#nameEdit {
            background: rgba(255,255,255,22);
            color: white;
            border: 2px solid rgba(102,240,200,120);
            border-radius: 12px;
            padding: 18px 20px;
            font-size: 18px;
        }
        QLineEdit#nameEdit::placeholder { color: rgba(255,255,255,170); }
        QLineEdit#nameEdit:focus {
            border: 2px solid rgba(102,240,200,220);
            background: rgba(255,255,255,28);
        }

        /* next button */
        QPushButton#nextBtn {
            color: white;
            font-size: 20px;
            font-weight: 800;
            border: none;
            border-radius: 14px;
            padding: 16px 24px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #2f78ff, stop:1 #18b88f);
        }
        QPushButton#nextBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3a84ff, stop:1 #1cc59a);
        }
        QPushButton#nextBtn:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #2a6de6, stop:1 #159f7f);
        }
        """

    def on_enter(self, ctx):
        self.ctx = ctx

        # 기존 cam 정지
        try:
            if hasattr(self.ctx, "cam") and self.ctx.cam:
                self.ctx.cam.stop()
        except Exception:
            pass

        # 얼굴 스트림 시작
        try:
            if hasattr(self.ctx, "face"):
                self.ctx.face.start_stream()
        except Exception:
            pass

        self._reset_ui()
        self._timer.start(30)

    def on_leave(self, _):
        self.vkbd.hide()
        self._end_collection()
        try:
            if self.ctx and hasattr(self.ctx, "face"):
                self.ctx.face.stop_stream()
        except Exception:
            pass

    # ---------------- 내부 로직 ----------------
    def _reset_ui(self):
        self.vkbd.hide()
        self._current_name = ""
        self.nameEdit.clear()
        self.nextBtn.setEnabled(True)
        self.collecting = False
        self.collected.clear()
        self.stage = 1
        self._render_stage()
        self.cameraLabel.clear()
        self._last_qimg = None

    def _end_collection(self, stop_timer: bool = True):
        self.collecting = False
        self.nextBtn.setText("얼굴 인식 시작")
        self.nextBtn.setEnabled(True)
        if stop_timer and self._timer.isActive():
            self._timer.stop()

    def _render_frame(self, qimg: QImage):
        self._last_qimg = qimg
        tw, th = max(1, self.cameraLabel.width()), max(1, self.cameraLabel.height())
        p = QPixmap.fromImage(qimg).scaled(
            tw, th, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        if p.width() > tw or p.height() > th:
            x = max(0, (p.width() - tw) // 2)
            y = max(0, (p.height() - th) // 2)
            p = p.copy(x, y, tw, th)
        self.cameraLabel.setPixmap(p)

    def _tick(self):
        if self.stage != 2 or not self.ctx:
            return

        fr_bgr = None
        try:
            sb = getattr(self.ctx.face, "backend", None)
            st = getattr(sb, "stream", None) if sb else None
            if st:
                fr_bgr, _faces, _ = st.read(timeout=0.0)
        except Exception:
            fr_bgr = None

        if fr_bgr is None:
            return

        rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self._render_frame(qimg)

        # 수집 중이면 임베딩 추출/누적
        if not self.collecting:
            return

        emb = None
        try:
            emb = self.ctx.face.detect_and_embed(fr_bgr)
        except Exception:
            emb = None

        if emb is not None:
            self.collected.append(emb)
            self.hint.setText(
                f"정면을 바라보고 자연스럽게 움직여 주세요 (수집 {len(self.collected)}/{self.target_n})"
            )

        if len(self.collected) >= self.target_n:
            try:
                self.ctx.face.add_user_samples(self._current_name, self.collected)
                self.hint.setText(f"{self._current_name} 등록이 완료되었습니다.")
            except Exception as e:
                self.hint.setText(f"저장 실패: {e}")
            finally:
                self._end_collection(stop_timer=False)
                try:
                    if self.ctx and hasattr(self.ctx, "router"):
                        self.ctx.router.navigate("start")
                except Exception:
                    pass
                self.finished.emit()
