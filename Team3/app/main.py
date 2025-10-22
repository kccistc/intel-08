# main.py
import sys
import os 
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QShortcut, QKeySequence, QFontDatabase, QFont
from PySide6.QtCore import Qt, QLocale
from core.context import AppContext
from core.router import Router

from views.start_page import StartPage
from views.exercise_page import ExercisePage
from views.guide_page import GuidePage
from views.summary_page import SummaryPage
from views.enroll_page import EnrollPage
from views.info_page import InfoPage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_app_font(font_path: str, point_size: int = 12) -> bool:
    fid = QFontDatabase.addApplicationFont(font_path)
    if fid == -1:
        return False
        
    fams = QFontDatabase.applicationFontFamilies(fid)
    if not fams:
        return False

    app_font = QFont(fams[0])
    app_font.setPointSize(point_size)
    app_font.setHintingPreference(QFont.PreferFullHinting)
    QApplication.setFont(app_font)
    return True

class MainWindow(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("자세어때")
        self.router = Router(ctx, parent=self)
        self.setCentralWidget(self.router)
        self.ctx.set_router(self.router)

        self.router.register("start", lambda: StartPage())
        self.router.register("guide", lambda: GuidePage())
        self.router.register("exercise", lambda: ExercisePage())
        self.router.register("summary", lambda: SummaryPage())
        self.router.register("enroll", lambda: EnrollPage())
        self.router.register("info",   lambda: InfoPage())

        QShortcut(QKeySequence(Qt.Key_F1), self).activated.connect(lambda: self.router.navigate("start"))
        QShortcut(QKeySequence(Qt.Key_F2), self).activated.connect(lambda: self.router.navigate("guide"))
        QShortcut(QKeySequence(Qt.Key_F3), self).activated.connect(lambda: self.router.navigate("exercise"))
        QShortcut(QKeySequence(Qt.Key_F4), self).activated.connect(lambda: self.router.navigate("info"))

        QShortcut(QKeySequence(Qt.Key_F11), self).activated.connect(self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_F12), self).activated.connect(QApplication.instance().quit)

        self.resize(1280, 720)
        self.router.navigate("start")

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()


if __name__ == "__main__":
    os.environ.setdefault("QT_MEDIA_BACKEND", "gstreamer") 
    os.environ.setdefault("GST_PLUGIN_FEATURE_RANK", "pulsesink:0")
    QLocale.setDefault(QLocale(QLocale.Korean, QLocale.SouthKorea))
    
    app = QApplication(sys.argv)
    
    font_path = os.path.join(BASE_DIR, "assets", "fonts", "GodoB.ttf")
    set_app_font(font_path, 15)

    ctx = AppContext()
    win = MainWindow(ctx)
    win.showFullScreen()  
    sys.exit(app.exec())