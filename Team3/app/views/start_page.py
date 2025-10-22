import os
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStackedLayout,
    QFrame, QSizePolicy, QGraphicsDropShadowEffect
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer, QSize, QRect, Signal
from core.page_base import PageBase

# ---------------- UI Util ----------------
def asset_path(*parts) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, *parts)

def crop_transparent_borders(pix: QPixmap) -> QPixmap:
    if pix.isNull():
        return pix
    img = pix.toImage().convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    left, right, top, bottom = w, 0, h, 0
    found = False
    for y in range(h):
        for x in range(w):
            if img.pixelColor(x, y).alpha() > 10:
                found = True
                if x < left: left = x
                if x > right: right = x
                if y < top: top = y
                if y > bottom: bottom = y
    if not found:
        return pix
    rect = QRect(left, top, right - left + 1, bottom - top + 1)
    return QPixmap.fromImage(img.copy(rect))

class StartPage(PageBase):
    registerRequested = Signal()

    def __init__(self):
        super().__init__()

        # --------- State ---------
        self.setMinimumSize(1, 1)
        self._bg_pix = None
        self._icon_cropped = None
        self._face_pix = None   

        # Auto-login params
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(1000)
        self._auto_timer.timeout.connect(self._auto_login_tick)
        self._hit_consecutive = 0
        self._need_hits = 2
        self._th_sim = 0.50

        # --------- Background (Static Image) ---------
        self.bg = QLabel(self)
        self.bg.setScaledContents(False)
        self.bg.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.bg.lower()  # 항상 맨 뒤

        # --------- Foreground (Glass Panel UI) ----------
        overlay = self._build_overlay()

        # Stack (bg image + overlay)
        stack = QStackedLayout(self)
        stack.setStackingMode(QStackedLayout.StackAll)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(self.bg)
        stack.addWidget(overlay)
        overlay.raise_()

        # 버튼/배경 에셋 로드
        self._try_load_assets()
        self._rescale_bg()  # 최초 1회 적용
        self._layout_mid_face()  # face.png 첫 반영

        # 초기 상태 문구
        self.set_status("등록되지 않은 회원입니다")

    # ========= UI 구성 =========
    def _build_overlay(self) -> QWidget:
        overlay = QWidget(self)
        overlay.setObjectName("StartPageRoot")

        root = QVBoxLayout(overlay)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(0)
        root.addStretch(1)

        self.panel = self._make_glass_panel()
        root.addWidget(self.panel, alignment=Qt.AlignCenter)
        root.addStretch(1)

        overlay.setStyleSheet(self._stylesheet())
        return overlay

    def _make_glass_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("glassPanel")
        panel.setAttribute(Qt.WA_StyledBackground, True)

        shadow = QGraphicsDropShadowEffect(panel)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 16)
        shadow.setColor(Qt.black)
        panel.setGraphicsEffect(shadow)

        self.panel_layout = QVBoxLayout(panel)
        self.panel_layout.setContentsMargins(72, 48, 72, 48)
        self.panel_layout.setSpacing(0)

        # ---- Top: Title + Subtitle
        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(0)

        self.title = QLabel("자세어때")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignHCenter)

        self.subtitle = QLabel("AI 기반 운동 분석 시스템")
        self.subtitle.setObjectName("Subtitle")
        self.subtitle.setAlignment(Qt.AlignHCenter)

        top_lay.addWidget(self.title)
        top_lay.addSpacing(6)
        top_lay.addWidget(self.subtitle)

        # ---- Mid: Face area (아이콘 자리/가이드)
        mid = QWidget()
        self.mid_lay = QVBoxLayout(mid)
        self.mid_lay.setContentsMargins(0, 0, 0, 0)
        self.mid_lay.setSpacing(0)

        self.face_wrap = QFrame()
        self.face_wrap.setObjectName("FaceWrap")
        face_row = QHBoxLayout(self.face_wrap)
        face_row.setContentsMargins(0, 0, 0, 0)
        face_row.setSpacing(0)

        self.face_icon = QLabel("")
        self.face_icon.setObjectName("FaceIcon")
        self.face_icon.setAlignment(Qt.AlignCenter)

        face_row.addWidget(self.face_icon)

        self.mid_lay.addStretch(1)
        self.mid_lay.addWidget(self.face_wrap, alignment=Qt.AlignHCenter)
        self.mid_lay.addSpacing(-10)
        self.mid_lay.addStretch(1)

        # ---- Bottom: Status + Button
        bottom = QWidget()
        bottom_lay = QVBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(0)

        self.status_label = QLabel("")
        self.lbl_status = self.status_label 
        self.status_label.setObjectName("Status")
        self.status_label.setAlignment(Qt.AlignHCenter)
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.status_label.setContentsMargins(64, 10, 64, 10)

        bottom_lay.addWidget(self.status_label, alignment=Qt.AlignHCenter)
        bottom_lay.addSpacing(28)

        self.btn_container = QFrame()
        self.btn_container.setObjectName("BtnContainer")
        self.btn_container.setAttribute(Qt.WA_StyledBackground, True)
        btn_container_lay = QVBoxLayout(self.btn_container)
        btn_container_lay.setContentsMargins(0, 0, 0, 0)
        btn_container_lay.setSpacing(0)

        self.btn_icon_label = QLabel(self.btn_container)
        self.btn_icon_label.setObjectName("BtnIconLabel")
        self.btn_icon_label.setAlignment(Qt.AlignCenter)
        self.btn_icon_label.setScaledContents(False)
        btn_container_lay.addWidget(self.btn_icon_label)

        self.btn_register = QPushButton(self.btn_container)
        self.btn_register.setObjectName("BtnRegister")
        self.btn_register.setCursor(Qt.PointingHandCursor)
        self.btn_register.setFlat(True)

        self.btn_overlay = QLabel(self.btn_container)
        self.btn_overlay.setObjectName("BtnOverlay")
        self.btn_overlay.hide()

        self.btn_register.pressed.connect(self.btn_overlay.show)
        self.btn_register.released.connect(self.btn_overlay.hide)
        self.btn_register.clicked.connect(lambda: self._goto("enroll"))

        bottom_lay.addWidget(self.btn_container, alignment=Qt.AlignHCenter)

        self.panel_layout.addWidget(top)
        self.panel_layout.addWidget(mid)
        self.panel_layout.setStretchFactor(mid, 1)
        self.panel_layout.addWidget(bottom)

        return panel

    def _stylesheet(self) -> str:
        return """
        #StartPageRoot { background: transparent; }
        #glassPanel {
            background: qlineargradient(x1:0 y1:0, x2:1 y2:1,
                stop:0 rgba(0,0,0,150),
                stop:1 rgba(255,255,255,30));
            border-radius: 32px;
            border: 1px solid rgba(255,255,255,42);
        }
        #Title {
            color: rgba(255, 165, 0, 0.9);
            font-size: 200px;
            font-weight: 1000;
            letter-spacing: 20px;
            margin: 0; padding: 0;
        }
        #Subtitle {
            color: rgba(179, 229, 252, 1);
            font-size: 40px;
            font-weight: 500;
            margin: 0; padding: 0;
        }
        #Status {
            color: #ffd27d;
            background: rgba(0,0,0,0.2);
            padding: 0px 100px;
            border-radius: 16px;
            font-size: 26px;
        }
        #FaceIcon {
            color: rgba(255,255,255,0.95);
            background: rgba(135, 206, 250, 0.35);
            border-radius: 100px;
        }
        #BtnIconLabel { background: transparent; }
        #BtnRegister  { background: transparent; border: none; padding: 0; margin: 0; }
        """

    # ========= Layout helpers =========
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_bg()
        self._size_panel()
        self._layout_mid_face()
        self._layout_status()
        self._layout_button()

    def _rescale_bg(self):
        # 정적 배경 이미지 전체 채우기 (비율 유지, 초과 부분 크롭)
        self.bg.setGeometry(self.rect())
        if self._bg_pix:
            scaled = self._bg_pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.bg.setPixmap(scaled)

    def _size_panel(self):
        target_w = int(self.width() * 0.90)
        target_h = int(self.height() * 0.90)
        target_w = max(720, target_w)
        target_h = max(540, target_h)
        self.panel.setFixedSize(target_w, target_h)

    def _layout_mid_face(self):
        face_size = 400
        self.face_wrap.setFixedSize(face_size, face_size)
        self.face_icon.setFixedSize(face_size, face_size)

        # face.png가 있으면 픽스맵을 사이즈에 맞게 스케일, 없으면 이모지 fallback
        if self._face_pix and not self._face_pix.isNull():
            target = self._face_pix.scaled(
                QSize(face_size, face_size),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.face_icon.setPixmap(target)
            self.face_icon.setText("")  # 텍스트 제거
        else:
            font_px = max(48, int(face_size * 0.45))
            self.face_icon.setPixmap(QPixmap())  # 픽스맵 제거
            self.face_icon.setText("👤")
            self.face_icon.setStyleSheet(f"#FaceIcon {{ font-size: {font_px}px; }}")

    def _layout_status(self):
        fm = self.status_label.fontMetrics()
        text = self.status_label.text()
        text_w = fm.horizontalAdvance(text)
        m = self.status_label.contentsMargins()
        content_w = text_w + m.left() + m.right()
        max_w = int(self.panel.width() * 0.80)
        fixed_w = min(content_w, max_w)
        self.status_label.setFixedWidth(fixed_w)
        self.status_label.adjustSize()

    def _layout_button(self):
        btn_w = int(self.panel.width() * 0.15)
        btn_h = max(80, int(btn_w / 4))
        self.btn_container.setFixedSize(btn_w, btn_h)

        container_radius = max(12, int(btn_h * 0.25))
        overlay_radius   = max(container_radius + 4, int(btn_h * 0.28))

        self.btn_container.setStyleSheet(
            f"#BtnContainer {{ background: transparent; border-radius: {container_radius}px; }}"
        )
        self.btn_overlay.setGeometry(0, 0, btn_w, btn_h)
        self.btn_overlay.setStyleSheet(
            f"background: rgba(0,0,0,0.18); border-radius: {overlay_radius}px;"
        )
        self.btn_register.setGeometry(0, 0, btn_w, btn_h)

        if self._icon_cropped and not self._icon_cropped.isNull():
            target = self._icon_cropped.scaled(
                QSize(btn_w, btn_h), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self.btn_icon_label.setPixmap(target)
        else:
            self.btn_icon_label.setText("회원가입")
            self.btn_icon_label.setAlignment(Qt.AlignCenter)
            self.btn_icon_label.setStyleSheet(
                "QLabel { color: white; font-size: 36px; font-weight: 800; background: transparent; }"
            )

    def _try_load_assets(self):
        # 배경
        bg_path = asset_path("assets", "background", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pm = QPixmap(bg_path)
            if not pm.isNull():
                self._bg_pix = pm

        # 버튼 아이콘 (투명 여백 크롭)
        btn_img = asset_path("assets", "buttons", "btn_register.png")
        if os.path.exists(btn_img):
            src = QPixmap(btn_img)
            if not src.isNull():
                self._icon_cropped = crop_transparent_borders(src)

        # 얼굴 아이콘 이미지
        face_img = asset_path("assets", "face.png")
        if os.path.exists(face_img):
            fp = QPixmap(face_img)
            if not fp.isNull():
                self._face_pix = fp

    # ========= Navigation helpers =========
    def _go_enroll(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate("enroll")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    # ========= Public helpers =========
    def set_status(self, text: str):
        self.status_label.setText(text)
        self._layout_status()

    # ========= Lifecycle =========
    def on_enter(self, ctx):
        self.ctx = ctx
        try:
            self.ctx.face.start_stream() 
        except Exception:
            pass
        self._hit_consecutive = 0
        self.set_status("카메라 준비 중… 정면을 바라봐 주세요.")
        self._auto_timer.start()

    def on_leave(self, ctx):
        self._auto_timer.stop()
        try:
            self.ctx.face.stop_stream()
        except Exception:
            pass

    # ========= Auto-login loop =========
    def _auto_login_tick(self):
        try:
            sb = getattr(self.ctx.face, "backend", None)
            st = getattr(sb, "stream", None) if sb else None
            running = bool(getattr(st, "_running", False))
            if not running:
                self.set_status("카메라 준비 중…")
                return
        except Exception:
            self.set_status("카메라 준비 중…")
            return

        emb = self.ctx.face.detect_and_embed(None)
        if emb is None:
            self._hit_consecutive = 0
            self.set_status("얼굴을 화면 중앙에 맞추고 정면을 바라봐 주세요.")
            return

        name, sim = self.ctx.face.match(emb, threshold=self._th_sim)
        if name:
            self._hit_consecutive += 1
            self.set_status(f"{name} 님 로그인 중...")
            if self._hit_consecutive >= self._need_hits:
                try:
                    from db.models import User
                    with self.ctx.SessionLocal() as s:
                        user = s.query(User).filter_by(name=name).one_or_none()
                        if user:
                            self.ctx.set_current_user(user.id, user.name)
                except Exception:
                    pass
                self._auto_timer.stop()
                self._goto("guide")
        else:
            self._hit_consecutive = 0
            self.set_status("등록되지 않은 얼굴이에요. ‘회원가입’을 눌러 등록해 주세요.")
