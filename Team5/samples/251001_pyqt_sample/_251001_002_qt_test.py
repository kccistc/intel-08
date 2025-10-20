import sys, random, time
from datetime import datetime, timezone
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QFont, QCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QPlainTextEdit, QLabel, QStackedLayout, QTableWidget,
    QTableWidgetItem, QFrame, QGridLayout, QHeaderView, QSizePolicy
)

# -----------------------------
# Utils
# -----------------------------
LOG_LEVELS = ["INFO", "DEBUG", "WARN", "ERROR"]
LOG_TOPICS = [
    "Net", "MQTT", "Anomaly", "Camera", "Pose", "DB", "Sensor", "TTS",
    "System", "Service", "Heartbeat"
]

SENSORS = [
    ("Temp", "°C"),
    ("Humidity", "%"),
    ("Roll", "°"),
    ("Pitch", "°"),
    ("AccelX", "m/s²"),
    ("AccelY", "m/s²"),
    ("AccelZ", "m/s²"),
]


def now_str():
    # return datetime.now().strftime("%H:%M:%S")
    """RFC3339 local time, e.g. 2025-10-01T10:23:45+09:00"""
    # return datetime.now().astimezone().isoformat(timespec="seconds")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------
# Simulated Server that emits random log lines
# -----------------------------
class FakeServer(QObject):
    log_generated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._emit_log)

    def start(self, interval_ms=800):
        self.timer.start(interval_ms)

    def stop(self):
        self.timer.stop()

    def _emit_log(self):
        lvl = random.choice(LOG_LEVELS)
        topic = random.choice(LOG_TOPICS)
        msg = random.choice([
            "OK", "connected", "disconnected", "restarting", "anomaly detected",
            "queue=12", "lat=37.4 lon=126.9", "fps=28.7", "db write ok",
            "timeout", "retry", "sensor drift", "calibrating"
        ])
        line = f"[{now_str()}] {lvl:<5} {topic:<9} :: {msg}"
        self.log_generated.emit(line)


# -----------------------------
# Log View Widget
# -----------------------------
class LogView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogView")

        grid = QGridLayout(self)
        grid.setContentsMargins(12, 12, 12, 12)
        # 3-column layout: 20 | center | 20
        grid.setColumnMinimumWidth(0, 20)
        grid.setColumnMinimumWidth(2, 20)
        grid.setColumnStretch(1, 1)  # center expands

        title = QLabel("Log Monitor")
        title.setFont(QFont("Inter", 20, QFont.Weight.Bold))
        grid.addWidget(title, 0, 1, Qt.AlignmentFlag.AlignCenter)

        self.tts_btn = QPushButton("Simulate TTS")
        self.tts_btn.setToolTip("Trigger a TTS command")
        grid.addWidget(self.tts_btn, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFrameShape(QFrame.Shape.Box)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMinimumSize(900, 420)  # ensure big enough center area
        self.text.setStyleSheet("QPlainTextEdit { background: #0f1115; color: #e6edf3; font-family: 'JetBrains Mono', monospace; font-size: 12pt; }")
        grid.addWidget(self.text, 2, 1)

        hint = QLabel("Click anywhere to toggle Sensor View ↔ Log View. Press 'T' to simulate TTS.")
        hint.setStyleSheet("color: #888;")
        grid.addWidget(hint, 3, 1, Qt.AlignmentFlag.AlignCenter)

    def append_log(self, line: str):
        self.text.appendPlainText(line)
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text.setTextCursor(cursor)


# -----------------------------
# Sensor View Widget
# -----------------------------
class SensorView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SensorView")

        grid = QGridLayout(self)
        grid.setContentsMargins(12, 12, 12, 12)
        # 3-column layout: 20 | center(stretch) | 20
        grid.setColumnMinimumWidth(0, 20)
        grid.setColumnMinimumWidth(2, 20)
        grid.setColumnStretch(1, 1)

        title = QLabel("Sensor Monitor")
        title.setFont(QFont("Inter", 20, QFont.Weight.Bold))
        grid.addWidget(title, 0, 1, Qt.AlignmentFlag.AlignCenter)

        # Table (center, large, no scrollbars)
        self.table = QTableWidget(len(SENSORS), 3)
        self.table.setHorizontalHeaderLabels(["Sensor", "Value", "Unit"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # Fill width/height and remove scrollbars
        hheader = self.table.horizontalHeader()
        hheader.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vheader = self.table.verticalHeader()
        vheader.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for row, (name, unit) in enumerate(SENSORS):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("-"))
            self.table.setItem(row, 2, QTableWidgetItem(unit))

        grid.addWidget(self.table, 1, 1)
        grid.setRowStretch(1, 1)  # make table consume available height

        hint = QLabel("Click anywhere to go back to Log View.")
        hint.setStyleSheet("color: #888;")
        grid.addWidget(hint, 2, 1, Qt.AlignmentFlag.AlignCenter)

        # Update timer for sensor values
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_values)

    def start_updates(self):
        self.timer.start(500)

    def stop_updates(self):
        self.timer.stop()

    def _update_values(self):
        for row, (name, unit) in enumerate(SENSORS):
            val = self._rand_value(name)
            self.table.setItem(row, 1, QTableWidgetItem(val))

    def _rand_value(self, name: str) -> str:
        if name == "Temp":
            return f"{random.uniform(18.0, 31.0):.1f}"
        if name == "Humidity":
            return f"{random.uniform(35, 85):.0f}"
        if name in ("Roll", "Pitch"):
            return f"{random.uniform(-15, 15):.1f}"
        if name.startswith("Accel"):
            return f"{random.uniform(-0.5, 0.5):.2f}"
        return f"{random.random():.3f}"

# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log Monitor + TTS + Sensor Toggle (Simulated)")
        self.setCursor(QCursor(Qt.CursorShape.BlankCursor))
        self.showFullScreen()

        self.central = QWidget()
        self.stack = QStackedLayout(self.central)
        self.setCentralWidget(self.central)

        self.log_view = LogView()
        self.sensor_view = SensorView()
        self.stack.addWidget(self.log_view)
        self.stack.addWidget(self.sensor_view)
        self.stack.setCurrentIndex(0)

        self.server = FakeServer()
        self.server.log_generated.connect(self.on_log)
        self.server.start(700)

        self.is_tts_active = False
        self.tts_overlay = QLabel("Speaking...")
        self.tts_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tts_overlay.setStyleSheet("QLabel { background: rgba(0,0,0,120); color: white; font-size: 28px; font-weight: 700; padding: 20px; border-radius: 12px; }")
        self.tts_overlay.setVisible(False)
        self._install_overlay()

        self.log_view.tts_btn.clicked.connect(self.trigger_tts)

        tts_act = QAction("TTS", self)
        tts_act.setShortcut("T")
        tts_act.triggered.connect(self.trigger_tts)
        self.addAction(tts_act)

        self._apply_base_styles()

    def _apply_base_styles(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #111317; }
            QLabel { color: #e6edf3; }
            QPushButton { padding: 6px 10px; border-radius: 8px; }
            QPushButton:hover { background: #2a2f3a; }
            QTableWidget { background: #0f1115; color: #e6edf3; gridline-color: #2a2f3a; }
            QHeaderView::section { background: #1a1f29; color: #c9d1d9; padding: 6px; }
            """
        )

    def _install_overlay(self):
        self.overlay_holder = QWidget(self.central)
        self.overlay_holder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(self.overlay_holder)
        layout.addStretch(1)
        layout.addWidget(self.tts_overlay, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.overlay_holder.setGeometry(self.central.rect())
        self.central.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.central and ev.type() == ev.Type.Resize:
            self.overlay_holder.setGeometry(self.central.rect())
        return super().eventFilter(obj, ev)

    def on_log(self, line: str):
        if self.stack.currentIndex() == 0:
            self.log_view.append_log(line)

    def trigger_tts(self):
        if self.is_tts_active:
            return
        self.is_tts_active = True

        self._central_prev_css = self.central.styleSheet()
        self.central.setStyleSheet("background: #2b3a20;")
        self.tts_overlay.setVisible(True)
        self.log_view.append_log(f"[{now_str()}] INFO  TTS       :: speaking started")

        QTimer.singleShot(2500, self._end_tts)

    def _end_tts(self):
        self.is_tts_active = False
        prev = getattr(self, "_central_prev_css", "")
        if prev:
            self.central.setStyleSheet(prev)
        else:
            self.central.setStyleSheet("background: #111317;")
        self.tts_overlay.setVisible(False)
        self.log_view.append_log(f"[{now_str()}] INFO  TTS       :: speaking finished")

    def mousePressEvent(self, ev):
        if self.stack.currentIndex() == 0:
            self.log_view.append_log(f"[{now_str()}] INFO  UI        :: touch -> open Sensor Monitor")
            self.log_view.append_log(f"[{now_str()}] INFO  DB        :: requesting sensor data...")
            self.stack.setCurrentIndex(1)
            QTimer.singleShot(400, self.sensor_view.start_updates)
        else:
            self.sensor_view.stop_updates()
            self.stack.setCurrentIndex(0)
            self.log_view.append_log(f"[{now_str()}] INFO  UI        :: touch -> back to Log View")
        super().mousePressEvent(ev)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
