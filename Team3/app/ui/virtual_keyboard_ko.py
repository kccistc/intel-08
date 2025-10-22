from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout,
    QSizePolicy, QLabel
)

S_BASE = 0xAC00
L_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
V_LIST = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
T_LIST = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

L_INDEX = {c:i for i,c in enumerate(L_LIST)}
V_INDEX = {c:i for i,c in enumerate(V_LIST)}
T_INDEX = {c:i for i,c in enumerate(T_LIST)}

V_COMB = {
    ('ㅗ','ㅏ'):'ㅘ', ('ㅗ','ㅐ'):'ㅙ', ('ㅗ','ㅣ'):'ㅚ',
    ('ㅜ','ㅓ'):'ㅝ', ('ㅜ','ㅔ'):'ㅞ', ('ㅜ','ㅣ'):'ㅟ',
    ('ㅡ','ㅣ'):'ㅢ'
}
T_COMB = {
    ('ㄱ','ㅅ'):'ㄳ',
    ('ㄴ','ㅈ'):'ㄵ', ('ㄴ','ㅎ'):'ㄶ',
    ('ㄹ','ㄱ'):'ㄺ', ('ㄹ','ㅁ'):'ㄻ', ('ㄹ','ㅂ'):'ㄼ', ('ㄹ','ㅅ'):'ㄽ', ('ㄹ','ㅌ'):'ㄾ', ('ㄹ','ㅍ'):'ㄿ', ('ㄹ','ㅎ'):'ㅀ',
    ('ㅂ','ㅅ'):'ㅄ'
}
# 복합받침 분해: (앞 음절에 남을 받침, 다음 음절로 넘어갈 초성)
T_SPLIT = {
    'ㄳ': ('ㄱ', 'ㅅ'), 'ㄵ': ('ㄴ', 'ㅈ'), 'ㄶ': ('ㄴ', 'ㅎ'),
    'ㄺ': ('ㄹ', 'ㄱ'), 'ㄻ': ('ㄹ', 'ㅁ'), 'ㄼ': ('ㄹ', 'ㅂ'),
    'ㄽ': ('ㄹ', 'ㅅ'), 'ㄾ': ('ㄹ', 'ㅌ'), 'ㄿ': ('ㄹ', 'ㅍ'),
    'ㅀ': ('ㄹ', 'ㅎ'), 'ㅄ': ('ㅂ', 'ㅅ'),
}

def compose(L, V, T):
    l = L_INDEX.get(L); v = V_INDEX.get(V); t = T_INDEX.get(T or '', 0)
    if l is None or v is None: return None
    return chr(S_BASE + (l * 21 + v) * 28 + t)

def decompose(ch):
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3): return None
    s = code - S_BASE
    l = s // 588; v = (s % 588) // 28; t = s % 28
    return (L_LIST[l], V_LIST[v], T_LIST[t] if T_LIST[t] else '')

# =================== Composer ===================
class HangulComposer:
    def __init__(self):
        self.L = self.V = self.T = ''
        self.buffer = []  # 확정된 문자열

    def _render_current(self) -> str:
        cur = ''
        if self.L and self.V: cur = compose(self.L, self.V, self.T) or ''
        elif self.L:          cur = self.L
        elif self.V:          cur = compose('ㅇ', self.V, '') or self.V
        return ''.join(self.buffer) + cur

    def _commit_running(self):
        if self.L and self.V:
            ch = compose(self.L, self.V, self.T)
            if ch: self.buffer.append(ch)
        elif self.L:
            self.buffer.append(self.L)
        elif self.V:
            ch = compose('ㅇ', self.V, '')
            if ch: self.buffer.append(ch)
        self.L = self.V = self.T = ''

    def expecting_final(self) -> bool:
        """종성 입력이 들어갈 타이밍인지 (초성,중성 O / 종성 X)."""
        return bool(self.L and self.V and not self.T)

    def input_char(self, ch: str) -> str:
        if ch == ' ':
            self._commit_running(); self.buffer.append(' '); return self._render_current()
        if ch == '\n':
            self._commit_running(); self.buffer.append('\n'); return self._render_current()

        is_vowel = ch in V_INDEX
        is_cons  = (ch in L_INDEX) or (ch in T_INDEX)

        if is_vowel:
            if not self.L and not self.V:
                self.L, self.V = 'ㅇ', ch; return self._render_current()
            if self.L and not self.V:
                self.V = ch; return self._render_current()
            if self.L and self.V and not self.T:
                comb = V_COMB.get((self.V, ch))
                if comb: self.V = comb; return self._render_current()
                self._commit_running(); self.L, self.V = 'ㅇ', ch; return self._render_current()
            if self.L and self.V and self.T:
                if self.T in T_SPLIT:
                    prev_t, next_l = T_SPLIT[self.T]
                    prev_ch = compose(self.L, self.V, prev_t)
                    if prev_ch: self.buffer.append(prev_ch)
                    self.L, self.V, self.T = next_l, ch, ''; return self._render_current()
                if self.T in L_INDEX:
                    prev_ch = compose(self.L, self.V, '')
                    if prev_ch: self.buffer.append(prev_ch)
                    self.L, self.V, self.T = self.T, ch, ''; return self._render_current()
                self._commit_running(); self.L, self.V = 'ㅇ', ch; return self._render_current()

        if is_cons:
            if not self.L:
                self.L = ch if ch in L_INDEX else 'ㅇ'; return self._render_current()
            if self.L and not self.V:
                self._commit_running(); self.L = ch if ch in L_INDEX else 'ㅇ'; return self._render_current()
            if self.L and self.V and not self.T:
                if ch in T_INDEX: self.T = ch; return self._render_current()
                self._commit_running(); self.L = ch if ch in L_INDEX else 'ㅇ'; return self._render_current()
            if self.L and self.V and self.T:
                comb = T_COMB.get((self.T, ch))
                if comb: self.T = comb; return self._render_current()
                self._commit_running(); self.L = ch if ch in L_INDEX else 'ㅇ'; return self._render_current()

        self._commit_running(); self.buffer.append(ch); return self._render_current()

    def backspace(self) -> str:
        if self.T:
            for a, b in T_COMB.items():
                if b == self.T: self.T = a[0]; return self._render_current()
            self.T = ''; return self._render_current()

        if self.V:
            for a, b in V_COMB.items():
                if b == self.V: self.V = a[0]; return self._render_current()
            self.V = ''; return self._render_current()

        if self.L:
            self.L = ''; return self._render_current()

        if not self.buffer: return self._render_current()

        last = self.buffer.pop()
        dc = decompose(last)
        if not dc: return self._render_current()

        Lc, Vc, Tc = dc
        if Tc: self.L, self.V, self.T = Lc, Vc, ''; return self._render_current()
        if Vc: self.L, self.V, self.T = Lc, '', ''; return self._render_current()
        self.L = self.V = self.T = ''; return self._render_current()

    def commit(self) -> str:
        self._commit_running(); return self._render_current()

# ---------- Virtual Keyboard (once-shift, docked) ----------
DOUBLE_MAP = {'ㄱ':'ㄲ','ㄷ':'ㄸ','ㅂ':'ㅃ','ㅈ':'ㅉ','ㅅ':'ㅆ'}

class VirtualKeyboardKO(QWidget):
    hiddenRequested = Signal()
    finished = Signal()
    keyPressed = Signal(str)

    def __init__(self, target_line_edit=None, parent=None):
        super().__init__(parent)
        self.setObjectName("VirtualKeyboardKO")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.target = target_line_edit
        self.composer = HangulComposer()
        self.shift_once = False
        self._build_ui()

    def setTarget(self, line_edit):
        self.target = line_edit
        self.composer = HangulComposer()
        if self.target:
            existing = self.target.text()
            if existing: self.composer.buffer = list(existing)

    # ----- UI -----
    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(16, 12, 16, 12); root.setSpacing(8)

        top = QHBoxLayout()
        title = QLabel("한글 키보드"); title.setObjectName("vkTitle")
        top.addWidget(title); top.addStretch(1)

        self.shiftBtn = QPushButton("Shift"); self.shiftBtn.setObjectName("vkShift")
        self.shiftBtn.setCheckable(False); self.shiftBtn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shiftBtn.pressed.connect(self._shift_once); top.addWidget(self.shiftBtn)

        hideBtn = QPushButton("숨기기"); hideBtn.setObjectName("vkHide")
        hideBtn.setFocusPolicy(Qt.FocusPolicy.NoFocus); hideBtn.pressed.connect(self._hide_self)
        top.addWidget(hideBtn)
        root.addLayout(top)

        grid = QGridLayout(); grid.setHorizontalSpacing(6); grid.setVerticalSpacing(6)

        row1 = list("ㅂㅈㄷㄱㅅㅛㅕㅑ")
        row2 = list("ㅁㄴㅇㄹㅎㅗㅓㅏㅣ")
        row3 = ['Shift','ㅋ','ㅌ','ㅊ','ㅍ','ㅠ','ㅜ','ㅡ','Back']
        row4 = ['Space','완료']

        self._place_row(grid, 0, row1)
        self._place_row(grid, 1, row2)
        self._place_row(grid, 2, row3, functional=True)
        self._place_row(grid, 3, row4, functional=True)
        root.addLayout(grid)

        self.setStyleSheet("""
        #VirtualKeyboardKO {
            background: rgba(20,20,24,235);
            border-top: 1px solid rgba(255,255,255,40);
        }
        #vkTitle { color: white; font-size: 14px; font-weight: 700; }
        QPushButton { padding: 6px 10px; }
        QPushButton[class="vkKey"] {
            color: white; background: rgba(255,255,255,18);
            border: 1px solid rgba(255,255,255,36); border-radius: 10px;
            font-size: 22px; font-weight: 800;
        }
        QPushButton[class="vkKey"]:hover { background: rgba(255,255,255,28); }
        QPushButton[class="vkFunc"] {
            color: white; background: rgba(255,255,255,26);
            border: 1px solid rgba(255,255,255,44); border-radius: 10px;
            font-size: 18px; font-weight: 800;
        }
        QPushButton#vkShift {
            color: white; background: rgba(255,255,255,32);
            border: 1px solid rgba(255,255,255,44);
            border-radius: 8px; padding: 6px 14px;
        }
        QPushButton#vkHide {
            color: white; background: rgba(255,255,255,18);
            border: 1px solid rgba(255,255,255,38); border-radius: 8px; padding: 6px 14px;
        }
        """)

    def _place_row(self, grid, r, keys, functional=False):
        c = 0
        for k in keys:
            if k == 'Space':
                btn = QPushButton('Space'); btn.setObjectName("vkSpace")
                btn.setProperty('class','vkFunc'); btn.setFixedHeight(64)
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                btn.pressed.connect(lambda ch=' ': self._press(ch))
                grid.addWidget(btn, r, c, 1, 5); c += 5; continue
            if k == '완료':
                btn = QPushButton('완료'); btn.setProperty('class','vkFunc'); btn.setFixedHeight(64)
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                btn.pressed.connect(self._enter)
                grid.addWidget(btn, r, c, 1, 2); c += 2; continue
            if k == 'Back':
                btn = QPushButton('←'); btn.setProperty('class','vkFunc'); btn.setFixedHeight(64)
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                btn.pressed.connect(self._back)
                grid.addWidget(btn, r, c, 1, 2); c += 2; continue
            if k == 'Shift':
                btn = self.shiftBtn; btn.setFixedHeight(64)
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                grid.addWidget(btn, r, c, 1, 2); c += 2; continue

            # 일반 자모 키는 clicked로 (중복/재진입 방지)
            btn = QPushButton(k); btn.setFixedHeight(64)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setProperty('class','vkKey'); btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda checked=False, ch=k: self._press(ch))
            grid.addWidget(btn, r, c); c += 1

    # ----- internals -----
    def _apply_text(self, text: str):
        if not self.target: return
        was_blocked = self.target.blockSignals(True)
        try:
            self.target.setText(text); self.target.setCursorPosition(len(text))
        finally:
            self.target.blockSignals(was_blocked)

    # ----- Actions -----
    def _shift_once(self):
        self.shift_once = True
        self.shiftBtn.setStyleSheet("background: rgba(80,140,255,0.45);")

    def _press(self, ch):
        if not self.target: return
        # 종성 후보면 쌍자음 치환 금지(Shift 유지)
        if self.shift_once and ch in DOUBLE_MAP and not self.composer.expecting_final():
            ch = DOUBLE_MAP[ch]
            self.shift_once = False
            self.shiftBtn.setStyleSheet("")
        text = self.composer.input_char(ch)
        self._apply_text(text)
        self.keyPressed.emit(ch)

    def _back(self):
        if not self.target: return
        text = self.composer.backspace()
        self._apply_text(text)
        if self.shift_once:
            self.shift_once = False; self.shiftBtn.setStyleSheet("")

    def _enter(self):
        if not self.target: return
        text = self.composer.commit()
        self._apply_text(text)
        self.finished.emit()
        self._hide_self()

    def _hide_self(self):
        self.hiddenRequested.emit(); self.hide()
