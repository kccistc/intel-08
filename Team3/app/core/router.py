from PySide6.QtWidgets import QStackedWidget

# 페이지 등록/전환 관리
class Router(QStackedWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._factories = {}
        self._instances = {}

    def register(self, name, factory):
        self._factories[name] = factory

    def navigate(self, name):
        # 인스턴스 없으면 생성
        page = self._instances.get(name)
        if page is None:
            page = self._factories[name]()
            self._instances[name] = page
            self.addWidget(page)

        # on_leave / on_enter 호출
        prev = self.currentWidget()
        if prev and hasattr(prev, "on_leave"):
            try: prev.on_leave(self.ctx)
            except Exception: pass

        self.setCurrentWidget(page)

        if hasattr(page, "on_enter"):
            try: page.on_enter(self.ctx)
            except Exception: pass

        return page
