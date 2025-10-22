from __future__ import annotations
import os, threading, queue
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from . import settings as S

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def _build_pipeline(cam: str, src_w: int, src_h: int, hef_path: str, post_so: str, post_func: str, cropper_so: str, io_mode: int = 2) -> str:
    return f"""
v4l2src device={cam} io-mode={io_mode} do-timestamp=true !
image/jpeg, width={src_w}, height={src_h} !
jpegparse ! avdec_mjpeg !
videoconvert ! videoscale !
video/x-raw,format=RGB,width={src_w},height={src_h} !
hailocropper name=crop so-path={cropper_so} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true
hailoaggregator name=agg
crop. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! agg.sink_0
crop. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 !
hailonet name=det hef-path={hef_path} batch-size=2 vdevice-group-id=1 force-writable=true !
queue max-size-buffers=5 max-size-bytes=0 !
hailofilter name=post so-path={post_so} function-name={post_func} qos=false !
queue max-size-buffers=5 max-size-bytes=0 ! agg.sink_1
agg. ! queue max-size-buffers=5 max-size-bytes=0 !
videoconvert ! video/x-raw,format=BGR !
appsink name=data_sink caps=video/x-raw,format=BGR emit-signals=true sync=false max-buffers=2 drop=true
"""

def _get_faces_from_buf(buf, w, h) -> List[Dict[str, Any]]:
    faces: List[Dict[str, Any]] = []
    try:
        import hailo
        roi = hailo.get_roi_from_buffer(buf)
        if not roi:
            return faces
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for d in dets:
            b = d.get_bbox()
            xmin, ymin = float(b.xmin()), float(b.ymin())
            bw, bh    = float(b.width()), float(b.height())

            k5 = None
            lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            if lms:
                pts = []
                for p in lms[0].get_points():
                    px = int((p.x()*bw + xmin) * w)
                    py = int((p.y()*bh + ymin) * h)
                    pts.append((px, py))
                if len(pts) >= 5:
                    k5 = pts[:5]

            faces.append({
                "bbox": [int(xmin*w), int(ymin*h), int((xmin+bw)*w), int((ymin+bh)*h)],
                "kpt5": k5
            })
    except Exception:
        pass
    return faces

def _bbox_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

class HailoFaceStream:
    def __init__(self,
                 hef_path: str,
                 post_so: str,
                 cropper_so: str,
                 cam: str,
                 src_size: Tuple[int,int] = (S.SRC_WIDTH, S.SRC_HEIGHT),
                 fps: int = S.SRC_FPS):
        self.hef_path = hef_path
        self.post_so  = post_so
        self.cropper_so = cropper_so
        self.cam = cam
        self.w, self.h = src_size
        self.fps = int(fps)

        self._loop: Optional[GLib.MainLoop] = None
        self._pipe = None
        self._appsink = None
        self._stop = threading.Event()
        self._running = False

        self._out_q: "queue.Queue[Tuple[Optional[np.ndarray], List[Dict[str,Any]], Tuple[int,int]]]" = queue.Queue(maxsize=1)

    def start(self):
        if self._running:
            return
        self._stop.clear()

        Gst.init(None)
        last_err = None
        self._pipe = None

        for m in (0, 2):
            try:
                desc = _build_pipeline(
                    cam=self.cam,
                    src_w=self.w,
                    src_h=self.h,
                    hef_path=self.hef_path,
                    post_so=self.post_so,
                    post_func=S.FACE_POST_FUNC,
                    cropper_so=self.cropper_so,
                    io_mode=m,
                )
                self._pipe = Gst.parse_launch(desc)
                break
            except Exception as e:
                last_err = e
                self._pipe = None

        if self._pipe is None:
            raise RuntimeError(f"HailoFaceStream: pipeline build failed: {last_err}")

        sink = self._pipe.get_by_name("data_sink")
        if sink is None:
            raise RuntimeError("HailoFaceStream: appsink 'data_sink' not found")

        self._appsink = sink
        try:
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
            sink.set_property("emit-signals", True)
        except Exception:
            pass
        sink.connect("new-sample", self._on_new_sample, None)

        self._loop = GLib.MainLoop()
        self._attach_bus_watch(self._loop, self._pipe)

        r = self._pipe.set_state(Gst.State.PLAYING)
        if r == Gst.StateChangeReturn.FAILURE:
            self._pipe.set_state(Gst.State.NULL)
            self._pipe = None
            raise RuntimeError("HailoFaceStream: pipeline start failed")

        threading.Thread(target=self._loop.run, daemon=True).start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        self._stop.set()
        try:
            if self._appsink is not None:
                self._appsink.set_property("emit-signals", False)
        except Exception:
            pass

        if self._pipe is not None:
            self._pipe.set_state(Gst.State.NULL)
            self._pipe = None

        self._appsink = None

        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:
                pass
            self._loop = None

        try:
            while True:
                self._out_q.get_nowait()
        except queue.Empty:
            pass

        self._running = False

    def _attach_bus_watch(self, loop, pipe):
        bus = pipe.get_bus()
        def on_msg(bus, msg):
            if msg.type == Gst.MessageType.ERROR:
                err, _dbg = msg.parse_error()
                print(f"[GST][face] ERROR: {err}", flush=True)
                self._stop.set()
                try: loop.quit()
                except Exception: pass
            return True
        bus.add_signal_watch()
        bus.connect("message", on_msg)

    def _on_new_sample(self, sink, _ud):
        if self._stop.is_set():
            return Gst.FlowReturn.EOS

        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()

        w, h = self.w, self.h
        try:
            caps = sample.get_caps()
            if caps and caps.get_size() > 0:
                s0 = caps.get_structure(0)
                w = int(s0.get_value('width'))
                h = int(s0.get_value('height'))
        except Exception:
            pass

        faces = _get_faces_from_buf(buf, w, h)
        if faces:
            faces.sort(key=lambda f: _bbox_area(f.get("bbox", [0,0,0,0])), reverse=True)
            faces = [faces[0]]

        bgr = None
        ok, mi = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                arr = np.frombuffer(mi.data, dtype=np.uint8)
                expected = w * h * 3
                if arr.size >= expected:
                    bgr = arr[:expected].reshape((h, w, 3)).copy()
            finally:
                buf.unmap(mi)

        try:
            while True:
                self._out_q.get_nowait()
        except queue.Empty:
            pass

        try:
            self._out_q.put_nowait((bgr, faces, (w, h)))
        except queue.Full:
            pass

        return Gst.FlowReturn.OK

    def read(self, timeout: Optional[float] = 0.0):
        import queue as q
        try:
            if timeout and timeout > 0:
                return self._out_q.get(timeout=timeout)
            else:
                return self._out_q.get_nowait()
        except q.Empty:
            return None, [], (self.w, self.h)
