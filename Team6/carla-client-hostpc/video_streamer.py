# File: video_streamer.py
import cv2
import numpy as np

class VideoStreamer:
    def __init__(self, target_ip, port, width, height, fps=30):
        """GStreamer 파이프라인을 설정하고 VideoWriter 객체를 생성합니다."""
        self.target_ip = target_ip
        self.port = port
        
        # H.264 인코딩 후 RTP 패킷으로 만들어 TCP로 전송하는 파이프라인
        pipeline = (
            f"appsrc ! "
            f"videoconvert ! "
            f"x264enc speed-preset=ultrafast tune=zerolatency bitrate=500 ! "
            f"rtph264pay ! "
            f"tcpserversink host={self.target_ip} port={self.port}"
        )
        
        # OpenCV의 VideoWriter를 이용해 GStreamer 파이프라인에 접근
        self.writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
        if not self.writer.isOpened():
            raise Exception(f"GStreamer VideoWriter on port {self.port} failed to open.")
        print(f"✅ Video streamer on port {self.port} is ready.")

    def send_frame(self, image_array):
        """CARLA 이미지(NumPy 배열)를 파이프라인으로 전송합니다."""
        self.writer.write(image_array)

    def release(self):
        """파이프라인을 해제합니다."""
        self.writer.release()