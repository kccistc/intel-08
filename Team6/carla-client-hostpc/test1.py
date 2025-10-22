import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import carla
import numpy as np
import cv2
import time

PI_IP = "10.10.14.183"
WIDTH, HEIGHT, FPS = 800, 320, 15

# GStreamer 초기화
Gst.init(None)

class CarlaCameraManager:
    def __init__(self, world, vehicle, width=WIDTH, height=HEIGHT, fps=FPS):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.fps = fps
        self.cameras = []
        self.latest_frames = {}

    def spawn_cameras(self):
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.width))
        cam_bp.set_attribute("image_size_y", str(self.height))
        cam_bp.set_attribute("sensor_tick", str(1.0 / self.fps))

        # 카메라 위치 (앞/뒤/좌/우/위/아래)
        positions = [
            (2.5 ,  0.0, 0.7, 0,   0,   0),   # 12시
            (2.5 ,  0.5, 0.7, 0,  30,   0),   # 2시
            (2.5 , -0.5, 0.7, 0, -30,   0),   # 10시
            (-2.5,  0.0, 0.7, 0, 180,   0),   # 6시
            (-2.5,  0.5, 0.7, 0, 150,   0),   # 4시
            (-2.5, -0.5, 0.7, 0,-150,   0),   # 8시
        ]

        for i, (x, y, z, pitch, yaw, roll) in enumerate(positions):
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
            )
            fov = 110 
            if i ==0 : 
                fov = 70
            cam_bp.set_attribute("fov", str(fov))# front를 제외한 나머지는 110 front는 70
            camera = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
            camera.listen(lambda image, cam_id=i: self._callback(image, cam_id))
            self.cameras.append(camera)

    def _callback(self, image, cam_id):
        if not self.cameras[cam_id].is_alive:
            return  # 이미 죽은 카메라라면 그냥 무시
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((self.height, self.width, 4))
        frame = arr[:, :, :3].copy()  # copy()로 별도 numpy 버퍼 확보

        # 최신 프레임 교체
        self.latest_frames[cam_id] = frame

        # ✅ CARLA 원본 image 객체 참조 제거 → UE 메모리 누수 방지
        del image , frame ,arr

    def destroy(self):
        for cam in self.cameras:
            if cam.is_alive:
                cam.stop()
                cam.destroy()
        self.cameras.clear()
        self.latest_frames.clear()


def make_pipeline(port, ip=PI_IP, width=WIDTH, height=HEIGHT, fps=FPS):
    pipeline_str = (
        "appsrc name=mysrc is-live=true block=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert ! "
        f"video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
        "x264enc tune=zerolatency bitrate=800 speed-preset=superfast key-int-max=30 ! "
        "rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={ip} port={port} sync=false async=false"
    )
    pipe = Gst.parse_launch(pipeline_str)
    appsrc = pipe.get_by_name("mysrc")

    # caps 명시
    caps = Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
    )
    appsrc.set_property("caps", caps)
    appsrc.set_property("is-live", True)
    appsrc.set_property("block", True)
    appsrc.set_property("do-timestamp", True)
    appsrc.set_property("max-bytes", width * height * 3 * 1)

    pipe.set_state(Gst.State.PLAYING)
    return pipe, appsrc


def push_frame_to_appsrc(frame, appsrc, frame_id=[0]):
    data = frame.tobytes()  # numpy → bytes (1회 복사)
    buf = Gst.Buffer.new_wrapped(data)  # 불필요한 추가 복사 없음

    # frame counter 기반 timestamp
    frame_id[0] += 1
    ts = frame_id[0] * Gst.SECOND // FPS
    buf.pts = ts
    buf.dts = ts
    buf.duration = Gst.SECOND // FPS

    ret = appsrc.emit("push-buffer", buf)
    if ret != Gst.FlowReturn.OK:
        print(f"⚠️ push-buffer failed: {ret}")
    
    del frame, data, buf, ts, ret



def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # === 동기 모드 설정 ===
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # === 차량 스폰 ===
    vehicle_bp = world.get_blueprint_library().filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle:
            print(f"✅ 차량 스폰 성공 @ {sp.location}")
            vehicle.set_autopilot(True)
            break
    if vehicle is None:
        print("❌ 차량 스폰 실패")
        return

    # === 카메라 스폰 ===
    cam_manager = CarlaCameraManager(world, vehicle)
    cam_manager.spawn_cameras()

    # === 6개 파이프라인 준비 ===
    pipelines, appsrcs = [], []
    for i in range(6):
        pipe, src = make_pipeline(5000 + i)
        pipelines.append(pipe)
        appsrcs.append(src)

    try:
        while True:
            world.tick()
            frames = cam_manager.latest_frames
            if len(frames) == 6:
                for i in range(6):
                    push_frame_to_appsrc(frames[i], appsrcs[i])

                # PC 미리보기
                row1 = np.hstack([frames[0], frames[1], frames[2]])
                row2 = np.hstack([frames[3], frames[4], frames[5]])
                grid = np.vstack([row1, row2])
                cv2.imshow("CARLA Preview", grid)
                
                if cv2.waitKey(1) == 27:
                    break

            del frames    
    finally:
        for p in pipelines:
            p.set_state(Gst.State.NULL)
        cam_manager.destroy()
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
