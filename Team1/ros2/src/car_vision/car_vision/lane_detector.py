import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from car_msgs.msg import LaneInfo # 이전에 정의한 메시지

class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector_node')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(LaneInfo, '/vision/lane_info', 10)
        self.bridge = CvBridge()
        self.get_logger().info('Lane Detector Node has been started.')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        lane_info_msg = LaneInfo()

        # =================================================================
        # 💻 TO-DO: OpenCV 담당자가 이 블록 내부를 채워주세요.
        # - 입력: cv_image (OpenCV 이미지)
        # - 처리: 차선 인식 알고리즘 수행
        # - 출력: lane_info_msg (LaneInfo 메시지)
        # =================================================================

        # 여기에 OpenCV 로직을 구현하고, 그 결과를 아래 메시지에 채워주세요.
        # (아래는 더미 데이터 예시입니다)
        lane_info_msg.is_detected = True
        lane_info_msg.curvature = 250.5
        lane_info_msg.offset = -0.15 # 차량이 중앙선에서 왼쪽으로 15cm 벗어남

        # =================================================================

        self.publisher_.publish(lane_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()