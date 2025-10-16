uint8 MSG_TYPE_EMERGENCY_BRAKE = 0
uint8 MSG_TYPE_OBSTACLE_AHEAD = 1
uint8 MSG_TYPE_RECKLESS_DRIVING = 2

uint8 msg_type       # 메시지 종류 (위의 타입 중 하나)
string vehicle_id    # 정보를 보낸 차량의 ID
float64 coordinateX     # 이벤트 발생 지점의 x좌표
float64 coordinateY     
