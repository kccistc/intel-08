from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import cv2

from .hailo_face_stream import HailoFaceStream
from . import settings as S

def _rec_forward_any(rec, rgb112):
    if hasattr(rec, "get_feat"):
        feat = rec.get_feat(rgb112)
    else:
        try:
            feat = rec.get(rgb112)
        except TypeError:
            class _DummyFace:
                bbox = None
                kps = None
            feat = rec.get(rgb112, _DummyFace())
    feat = np.asarray(feat, dtype=np.float32)
    n = float(np.linalg.norm(feat)) + 1e-9
    return feat / n

class FaceBackendBase:
    def detect_and_embed(self, bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        raise NotImplementedError
    def close(self) -> None: ...

_ALIGN_STD_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def _align_by_5pts(bgr: np.ndarray, kpt5: List[Tuple[int, int]], out_size: Tuple[int,int] = (112,112)) -> np.ndarray:
    src = np.array(kpt5, dtype=np.float32)
    dst = _ALIGN_STD_5PTS.copy()
    if out_size != (112,112):
        sx = out_size[0] / 112.0
        sy = out_size[1] / 112.0
        dst[:,0] *= sx; dst[:,1] *= sy
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        x = int(min(p[0] for p in kpt5)); y = int(min(p[1] for p in kpt5))
        X = int(max(p[0] for p in kpt5)); Y = int(max(p[1] for p in kpt5))
        x = max(0, x); y = max(0, y)
        crop = bgr[y:Y, x:X].copy() if (Y>y and X>x) else bgr
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

class HailoFaceBackend(FaceBackendBase):
    def __init__(self,
                 det_hef: Optional[str] = None,
                 post_so: Optional[str] = None,
                 cropper_so: Optional[str] = None,
                 cam: Optional[str] = None,
                 det_input_size: Tuple[int,int] = (S.SRC_WIDTH, S.SRC_HEIGHT),
                 face_app: str = "buffalo_l",
                 face_input: Tuple[int,int] = (112,112)):

        self.stream = HailoFaceStream(
            hef_path   = det_hef   or getattr(S, "FACE_DET_HEF", None),
            post_so    = post_so   or getattr(S, "FACE_POST_SO", None),
            cropper_so = cropper_so or getattr(S, "CROPPER_SO", None),
            cam        = cam       or getattr(S, "CAM", "/dev/video0"),
            src_size   = det_input_size
        )

        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(
            name=face_app,
            root=str(S.INSIGHTFACE_HOME),
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )

        self._app.prepare(ctx_id=-1, det_size=(640,640))
        self._rec = self._app.models.get('recognition', None)
        if self._rec is None:
            raise RuntimeError("Recognition model not loaded")
        self._face_in = face_input
        self._ok = True

        import numpy as np, cv2
        dummy = np.zeros((self._face_in[1], self._face_in[0], 3), np.uint8)
        _ = _rec_forward_any(self._rec, cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))

    @property
    def ok(self) -> bool:
        return bool(self._ok and (self.stream is not None) and (self._rec is not None))

    def start(self):
        if self.stream:
            self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()

    def detect_and_embed(self, bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        fr, faces, _ = self.stream.read(timeout=0.05)

        if fr is None and bgr is not None:
            fr = bgr
        if fr is None:
            return None

        if not faces:
            return None

        f0 = faces[0]
        kpt5 = f0.get("kpt5")
        if not kpt5 or len(kpt5) != 5:
            return None

        aligned = _align_by_5pts(fr, kpt5, out_size=self._face_in)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        emb = _rec_forward_any(self._rec, rgb)
        return emb

    def close(self):
        self.stop()
