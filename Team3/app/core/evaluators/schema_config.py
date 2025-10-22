from __future__ import annotations
import json, time, threading
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]  
ANGLE_JSON = ROOT / "data" / "angle_data.json"

class DynamicSchema:
    """
    JSON 설정을 읽어 섹션별(upper_body / lower_body / core) SCHEMA 제공.
    파일 mtime 변화를 감지해 런타임 중에도 자동 반영(핫 리로드).
    """
    def __init__(self, section: str, default_schema: Dict[str, Any], json_path: Path = ANGLE_JSON):
        self.section = section
        self.default_schema = default_schema
        self.json_path = Path(json_path)
        self._lock = threading.RLock()
        self._schema: Dict[str, Any] = default_schema
        self._last_mtime = -1.0
        self._last_check = 0.0
        self._throttle_sec = 2  # 리로드 체크 최소 주기

        self._try_reload(force=True)

    def _try_reload(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_check) < self._throttle_sec:
            return
        self._last_check = now

        try:
            if not self.json_path.exists():
                return
            mtime = self.json_path.stat().st_mtime
            if (not force) and (mtime == self._last_mtime):
                return

            with self.json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return

            section_data = data.get(self.section)
            if not isinstance(section_data, dict):
                return

            with self._lock:
                self._schema = section_data
                self._last_mtime = mtime
        except Exception:
            # 실패 시 이전/기본값 유지
            pass

    def reload_if_changed(self, throttle_sec: float = 2.0):
        with self._lock:
            self._throttle_sec = throttle_sec
        self._try_reload(force=False)

    def get_mode(self, mode: str) -> Dict[str, Any]:
        self.reload_if_changed()
        with self._lock:
            return self._schema.get(mode, self.default_schema.get(mode, {}))

    def full(self) -> Dict[str, Any]:
        self.reload_if_changed()
        with self._lock:
            return dict(self._schema)

_REGISTRY: Dict[str, DynamicSchema] = {}

def get_dynamic_schema(section: str, default_schema: Dict[str, Any]) -> DynamicSchema:
    if section not in _REGISTRY:
        _REGISTRY[section] = DynamicSchema(section=section, default_schema=default_schema)
    return _REGISTRY[section]
