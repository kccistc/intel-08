import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
import json
import socket
import struct
import threading
import hmac
import hashlib

def verify_hmac(raw_wo_sig:bytes, key:str, sig_hex:str)->bool:
    """HMAC-SHA256 서명을 검증하는 함수"""
    calc = hmac.new(key.encode("utf-8"), raw_wo_sig, hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, sig_hex)

def v2x_listener_thread(node: Node):
    """UDP 멀티캐스트를 수신하고, HMAC 검증 후 ROS2 토픽으로 발행하는 스레드"""
    mcast_group = "239.20.20.20"
    mcast_port = 5520
    
    # ROS2 파라미터에서 HMAC 키 가져오기
    hmac_key = node.get_parameter('hmac_key').get_parameter_value().string_value
    if hmac_key:
        node.get_logger().info('HMAC signature verification is enabled.')
    else:
        node.get_logger().warn('HMAC key is not set. Signature verification is disabled.')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', mcast_port))
    except OSError as e:
        node.get_logger().error(f"Failed to bind to port {mcast_port}: {e}")
        return # 스레드 종료

    mreq = struct.pack("4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    publisher = node.create_publisher(String, '/v2x/alert', 10)
    node.get_logger().info(f'V2X listener thread started, listening on {mcast_group}:{mcast_port}')

    while rclpy.ok():
        try:
            raw_data, addr = sock.recvfrom(2048)
            
            try:
                alert_obj = json.loads(raw_data.decode('utf-8'))
                
                # HMAC 서명 검증 로직
                if hmac_key and "sig" in alert_obj:
                    sig = alert_obj["sig"].get("value", "")
                    # 'sig' 필드를 제외한 나머지 부분으로 검증용 데이터 재생성
                    clone = dict(alert_obj)
                    clone.pop("sig", None)
                    raw_wo_sig = json.dumps(clone, separators=(',',':')).encode('utf-8')
                    
                    if not verify_hmac(raw_wo_sig, hmac_key, sig):
                        node.get_logger().warn(f"HMAC verification failed for message from {addr[0]}. Dropping.")
                        continue # 서명 검증 실패 시 메시지 무시
                
                # 검증 통과 또는 검증 비활성화 시 메시지 발행
                msg = String()
                msg.data = raw_data.decode('utf-8')
                publisher.publish(msg)

            except json.JSONDecodeError:
                node.get_logger().warn(f"Failed to decode JSON from V2X message: {raw_data.decode('utf-8', errors='ignore')}")
                continue
            except Exception as e:
                node.get_logger().error(f"Error during HMAC verification: {e}")

        except Exception as e:
            if rclpy.ok():
                node.get_logger().error(f"V2X listener socket error: {e}")
    sock.close()

class V2XSubscriberNode(Node):
    def __init__(self):
        super().__init__('v2x_subscriber_node')
        
        # HMAC 키를 위한 ROS2 파라미터 선언 (기본값: 빈 문자열)
        self.declare_parameter('hmac_key', '')

        # 별도의 스레드에서 V2X UDP 리스너 실행
        self.listener_thread = threading.Thread(target=v2x_listener_thread, args=(self,), daemon=True)
        self.listener_thread.start()
        
        self.get_logger().info('V2X Subscriber node has been started.')

def main(args=None):
    rclpy.init(args=args)
    v2x_subscriber_node = V2XSubscriberNode()
    rclpy.spin(v2x_subscriber_node)
    v2x_subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()