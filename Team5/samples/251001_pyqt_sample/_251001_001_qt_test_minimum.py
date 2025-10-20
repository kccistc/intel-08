# PyQt6 버전
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Quickstart")
        self.resize(360, 180)

        self.label = QLabel("Hello, PyQt!", self)
        self.btn = QPushButton("Click me", self)
        self.btn.clicked.connect(self.on_click)  # 시그널 → 슬롯

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.btn)

    def on_click(self):
        self.label.setText("Button clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())
