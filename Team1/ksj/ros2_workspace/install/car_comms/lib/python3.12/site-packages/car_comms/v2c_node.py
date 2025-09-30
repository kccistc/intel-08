import rclpy
from rclpy.node import Node
from car_msgs.msg import V2VAlert # 기존 메시지 재활용 또는 새 메시지 정의
import paho.mqtt.client as mqtt

# --- MQTT 설정 ---
MQTT_BROKER_IP = "YOUR_MQTT_BROKER_IP" # 예: "192.168.1.100"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_UPLINK = "vehicles/car_01/status"   # 차량 -> 센터로 보낼 토픽
MQTT_TOPIC_DOWNLINK = "vehicles/car_01/command" # 센터 -> 차량으로 받을 토픽

class V2CNode(Node):
    def __init__(self):
        super().__init__('v2c_node')
        self.publisher_ = self.create_publisher(V2VAlert, '/v2c/alert', 10)
        
        # MQTT 클라이언트 설정
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        try:
            self.mqtt_client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)
            self.mqtt_client.loop_start()
            self.get_logger().info(f'MQTT client connected to {MQTT_BROKER_IP}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to MQTT broker: {e}')

        # 주기적으로 차량 상태를 발행(Publish)하기 위한 타이머
        self.uplink_timer = self.create_timer(5.0, self.publish_vehicle_status)

    def on_mqtt_connect(self, client, userdata, flags, rc, properties):
        self.get_logger().info(f"Connected with result code {rc}")
        # 연결 성공 시, 서버로부터 명령을 받을 토픽 구독
        client.subscribe(MQTT_TOPIC_DOWNLINK)
        self.get_logger().info(f"Subscribed to MQTT topic: {MQTT_TOPIC_DOWNLINK}")

    def on_mqtt_message(self, client, userdata, msg):
        # MQTT 메시지(명령)를 받으면 이 함수가 호출됨
        try:
            payload = msg.payload.decode()
            self.get_logger().info(f"Received MQTT message: {payload}")
            ros_alert_msg = V2VAlert()

            # =================================================================
            # 💻 TO-DO: V2C 담당자가 이 블록 내부를 채워주세요.
            # - 처리: 수신한 MQTT 메시지(payload)를 파싱하여 ROS2 메시지로 변환
            # - 출력: ros_alert_msg (V2VAlert 메시지)
            # =================================================================
            
            # 예시: 서버에서 "EMERGENCY_BRAKE" 메시지를 받으면 ROS2에 경고 발행
            if "EMERGENCY_BRAKE" in payload:
                ros_alert_msg.msg_type = V2VAlert.MSG_TYPE_EMERGENCY_BRAKE
                ros_alert_msg.distance = 100.0 # 예시 거리
                self.publisher_.publish(ros_alert_msg)
                self.get_logger().warn('Published V2C Alert to ROS system!')

            # =================================================================

        except Exception as e:
            self.get_logger().error(f"Error processing MQTT message: {e}")

    def publish_vehicle_status(self):
        # =================================================================
        # 💻 TO-DO: V2C 담당자가 이 블록 내부를 채워주세요.
        # - 처리: 차량의 현재 상태(위치, 속도 등)를 MQTT로 서버에 전송
        # =================================================================
        
        # 이 부분은 다른 ROS2 노드(예: state_estimator)의 정보를 구독해서
        # 실제 차량 데이터를 전송하도록 확장해야 합니다.
        status_payload = '{"location": "37.5665, 126.9780", "speed": 45.5}'
        self.mqtt_client.publish(MQTT_TOPIC_UPLINK, status_payload)
        self.get_logger().info(f"Published status to MQTT: {status_payload}")
        
        # =================================================================

def main(args=None):
    rclpy.init(args=args)
    node = V2CNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()