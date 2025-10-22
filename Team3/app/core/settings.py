from __future__ import annotations
import os
from pathlib import Path

APP_ROOT   = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_ROOT / "models"

# 카메라 
CAM = os.environ.get(
    "CAM",
    # "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH1901_20240311_v007-video-index0"
    "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0"
)

SRC_WIDTH  = int(os.environ.get("SRC_WIDTH",  "1280"))
SRC_HEIGHT = int(os.environ.get("SRC_HEIGHT", "720"))
SRC_FPS    = int(os.environ.get("SRC_FPS",    "30"))

# 포즈(Hailo YOLOv8-Pose) 
HEF       = os.environ.get("HEF", str(MODELS_DIR / "yolov8s_pose.hef"))
POST_SO   = os.environ.get("POST_SO", str(MODELS_DIR / "libyolov8pose_postprocess.so"))
POST_FUNC = os.environ.get("POST_FUNC", "filter_letterbox")

# cropper는 포즈/얼굴 모두에서 재사용 가능
CROPPER_SO = os.environ.get(
    "CROPPER_SO",
    "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"
)

# 동작 인식 TCN
TCN_ONNX = os.environ.get("TCN_ONNX", str(MODELS_DIR / "tcn.onnx"))
TCN_JSON = os.environ.get("TCN_JSON", str(MODELS_DIR / "tcn.json"))

# InsightFace 설정
FACE_APP_NAME  = os.environ.get("FACE_APP_NAME", "buffalo_l")
FACE_DET_SIZE  = tuple(map(int, os.environ.get("FACE_DET_SIZE", "640,640").split(",")))
FACE_INPUT_HW  = tuple(map(int, os.environ.get("FACE_INPUT_HW", "112,112").split(","))) 
FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.40"))

INSIGHTFACE_HOME = Path(os.environ.get("INSIGHTFACE_HOME",str(MODELS_DIR / "insightface_cache")))

USE_HAILO_FACE = True            # Hailo 검출 사용

# --- Hailo 얼굴 검출/임베딩 리소스 ---
FACE_DET_HEF   = str(MODELS_DIR / "retinaface_mobilenet_v1.hef")
FACE_POST_SO   = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libface_detection_post.so"
FACE_POST_FUNC = "retinaface"    

WINDOW_TITLE   = os.environ.get("WINDOW_TITLE", "Hailo YOLOv8-Pose")
FULLSCREEN     = bool(int(os.environ.get("FULLSCREEN", "0")))
WINDOW_SIZE    = (SRC_WIDTH, SRC_HEIGHT)
SHOW_INFO_BAR  = bool(int(os.environ.get("SHOW_INFO_BAR", "0")))