from __future__ import annotations
import numpy as np
from typing import List, Optional
from db.models import User, FaceEmbedding
from . import settings as S
from .face_backends import FaceBackendBase, HailoFaceBackend

class FaceService:
    def __init__(self, SessionLocal, backend: FaceBackendBase | None = None):
        self.SessionLocal = SessionLocal
        self.backend: Optional[FaceBackendBase] = backend
        self._enabled = False

        try:
            if self.backend is None:
                self.backend = HailoFaceBackend(
                    det_hef=getattr(S, "FACE_DET_HEF", None),
                    post_so=getattr(S, "FACE_POST_SO", None),
                    cropper_so=getattr(S, "CROPPER_SO", None),
                    cam=getattr(S, "CAM", None),
                    det_input_size=(S.SRC_WIDTH, S.SRC_HEIGHT),
                    face_app=S.FACE_APP_NAME,
                    face_input=getattr(S, "FACE_INPUT_HW", (112, 112)),
                )
            self._enabled = True
        except Exception:
            self.backend = None
            self._enabled = False

        self._cache = []
        self._rebuild_cache()

    def start_stream(self):
        if self.enabled and self.backend:
            try: self.backend.start()
            except Exception: pass

    def stop_stream(self):
        if self.backend:
            try: self.backend.stop()
            except Exception: pass

    @property
    def enabled(self) -> bool:
        return bool(getattr(self, "_enabled", False) and self.backend is not None)

    def _rebuild_cache(self) -> None:
        self._cache.clear()
        with self.SessionLocal() as s:
            rows = (
                s.query(FaceEmbedding, User)
                 .join(User, FaceEmbedding.user_id == User.id)
                 .all()
            )
            for fe, user in rows:
                emb = np.frombuffer(fe.embedding, dtype=np.float32)
                n = float(np.linalg.norm(emb)) + 1e-9
                self._cache.append((int(user.id), str(user.name), emb / n))

    def detect_and_embed(self, bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        try:
            return self.backend.detect_and_embed(bgr)
        except Exception:
            return None

    def add_user_samples(self, name: str, embeddings: List[np.ndarray]) -> int:
        safe = (name or "").strip()
        if not safe:
            raise ValueError("빈 이름은 등록할 수 없습니다.")
        if not embeddings:
            raise ValueError("임베딩이 비어 있습니다.")

        with self.SessionLocal() as s:
            user = User(name=safe)
            s.add(user)
            s.flush()

            for emb in embeddings:
                emb = np.asarray(emb, dtype=np.float32)
                s.add(FaceEmbedding(user_id=user.id, dim=int(emb.size), embedding=emb.tobytes()))

            s.commit()
            uid = int(user.id)

        self._rebuild_cache()
        return uid

    def match(self, emb: np.ndarray, threshold: float = S.FACE_MATCH_THRESHOLD) -> tuple[Optional[str], float]:
        if not self._cache:
            return None, 0.0

        q = np.asarray(emb, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) + 1e-9)

        best_name, best_sim = None, -1.0
        for _uid, name, ref in self._cache:
            sim = float(np.dot(q, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim

        return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)

    def close(self) -> None:
        try:
            if self.backend:
                self.backend.close()
        except Exception:
            pass
