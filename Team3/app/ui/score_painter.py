from PySide6.QtWidgets import QWidget, QLabel, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PySide6.QtGui import QColor

class ScoreOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def resizeEvent(self, e):
        self.setGeometry(self.parent().rect())
        return super().resizeEvent(e)

    def show_score(self, text: str = "100", base_px: int | None = None,
                   text_qcolor: QColor | None = None, 
                   supertext: str | None = None, super_qcolor: QColor | None = None):
        lbl = QLabel(text, self)
        lbl.setAttribute(Qt.WA_TranslucentBackground, True)
        lbl.setAlignment(Qt.AlignCenter)

        px = base_px if base_px else max(160, int(self.width() * 0.15))
        tc = text_qcolor or QColor(0, 128, 255)
        tc_r, tc_g, tc_b, tc_a = tc.red(), tc.green(), tc.blue(), tc.alpha()
        lbl.setStyleSheet(f"""
            QLabel {{
                color: rgba({tc.red()},{tc.green()},{tc.blue()},{tc.alpha()});
                font: 1000 {px}px "Pretendard";
                padding: 8px 16px;
                background: rgba(0,0,0,70);        
                border-radius: 12px;
                border: 2px solid rgba(0,0,0,120); 
            }}
        """)
        lbl.adjustSize()

        x = (self.width() - lbl.width()) // 2
        y = (self.height() - lbl.height()) // 2 - int(self.height() * 0.1)
        lbl.move(x, y)
        lbl.show()

        top_lbl = None
        top_fx = None
        if supertext:
            top_lbl = QLabel(supertext, self)
            top_lbl.setAttribute(Qt.WA_TranslucentBackground, True)
            top_lbl.setAlignment(Qt.AlignCenter)

            stc = super_qcolor or tc
            stc_r, stc_g, stc_b, stc_a = stc.red(), stc.green(), stc.blue(), stc.alpha()

            top_px = max(18, int(px * 0.28))
            top_lbl.setStyleSheet(f"""
                QLabel {{
                    color: rgba({stc_r},{stc_g},{stc_b},{stc_a});
                    font: 800 {top_px}px "Pretendard";
                    padding: 4px 10px;
                    background: rgba(0,0,0,70);
                    border-radius: 10px;
                    border: 1px solid rgba(0,0,0,120);
                }}
            """)
            top_lbl.adjustSize()
            top_x = (self.width() - top_lbl.width()) // 2
            top_y = y - top_lbl.height() - 12
            top_lbl.move(top_x, top_y)
            top_lbl.show()

            top_fx = QGraphicsOpacityEffect(top_lbl)
            top_lbl.setGraphicsEffect(top_fx)
            top_fx.setOpacity(0.0)

        # 투명도 효과
        fx = QGraphicsOpacityEffect(lbl)
        lbl.setGraphicsEffect(fx)
        fx.setOpacity(0.0) 

        # 1) 팝(등장) 연출: 0.0→1.0로 페이드인 + 가벼운 '커짐' 효과
        fade_in = QPropertyAnimation(fx, b"opacity", self)
        fade_in.setDuration(120)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.OutCubic)

        # 폰트 사이즈 살짝 0.85x → 1.0x로 키우기 (QVariantAnimation로 간단 스케일)
        start_px = int(px * 0.85)
        pop_anim = QVariantAnimation(self)
        pop_anim.setDuration(120)
        pop_anim.setStartValue(start_px)
        pop_anim.setEndValue(px)
        pop_anim.setEasingCurve(QEasingCurve.OutBack)

        def _apply_font(v):
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: rgba({tc_r},{tc_g},{tc_b},{tc_a});
                    font: 1000 {int(v)}px "Pretendard";
                    padding: 8px 16px;
                    background: rgba(0,0,0,70);
                    border-radius: 12px;
                    border: 2px solid rgba(0,0,0,120);
                }}
            """)
            # 크기 바뀌면 가운데 유지
            lbl.adjustSize()
            new_x = (self.width() - lbl.width()) // 2
            new_y = (self.height() - lbl.height()) // 2 - int(self.height() * 0.1)
            lbl.move(new_x, new_y)

            if supertext and top_lbl:
                top_lbl.adjustSize()
                top_x2 = (self.width() - top_lbl.width()) // 2
                top_y2 = new_y - top_lbl.height() - 12
                top_lbl.move(top_x2, top_y2)

        pop_anim.valueChanged.connect(_apply_font)

        # 2) 위로 떠오르며 사라짐: 위치 y-80, 불투명도 1→0
        move = QPropertyAnimation(lbl, b"pos", self)
        move.setDuration(900)
        move.setStartValue(QPoint(x, y))
        move.setEndValue(QPoint(x, y - 80))
        move.setEasingCurve(QEasingCurve.OutCubic)

        fade_out = QPropertyAnimation(fx, b"opacity", self)
        fade_out.setDuration(900)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.InQuad)

        if supertext and top_lbl and top_fx:
            top_fade_in = QPropertyAnimation(top_fx, b"opacity", self)
            top_fade_in.setDuration(120)
            top_fade_in.setStartValue(0.0)
            top_fade_in.setEndValue(1.0)
            top_fade_in.setEasingCurve(QEasingCurve.OutCubic)

            top_move = QPropertyAnimation(top_lbl, b"pos", self)
            top_move.setDuration(900)
            top_move.setStartValue(QPoint(top_lbl.x(), top_lbl.y()))
            top_move.setEndValue(QPoint(top_lbl.x(), top_lbl.y()))
            top_move.setEasingCurve(QEasingCurve.OutCubic)

            top_fade_out = QPropertyAnimation(top_fx, b"opacity", self)
            top_fade_out.setDuration(900)
            top_fade_out.setStartValue(1.0)
            top_fade_out.setEndValue(0.0)
            top_fade_out.setEasingCurve(QEasingCurve.InQuad)

            top_fade_in.start()
            top_move.start()
            top_fade_out.start()

        # 정리
        def cleanup():
            if supertext and top_lbl:
                top_lbl.deleteLater()
            lbl.deleteLater()
       
        fade_out.finished.connect(cleanup)

        # 시퀀스 실행: 팝(120ms) → 떠오르며 페이드(900ms)
        fade_in.start()
        pop_anim.start()
        move.start()
        fade_out.start()
