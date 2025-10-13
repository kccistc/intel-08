uint8 MSG_TYPE_EMERGENCY_BRAKE = 0
uint8 MSG_TYPE_OBSTACLE_AHEAD = 1
uint8 MSG_TYPE_RECKLESS_DRIVING = 2
uint8 MSG_TYPE_UNKNOWN = 255

# 메시지의 타임스탬프와 프레임 ID
std_msgs/Header header

# 메시지 종류 (위의 타입 상수 중 하나)
uint8 msg_type

# 정보를 보낸 차량의 ID
string vehicle_id

# 이벤트 발생 지점 좌표 (geometry_msgs/Point 타입 사용)
geometry_msgs/Point position

# 메시지의 신뢰도 (0.0 ~ 1.0)
float32 confidence_score