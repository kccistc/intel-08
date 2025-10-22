# 메시지 헤더 (타임스탬프 포함)
builtin_interfaces/Time stamp

# 이벤트를 감지한 CCTV의 ID
string camera_id

# 이벤트 종류 (예: "accident_detected")
string event_type

# 이벤트 발생 위치
geometry_msgs/Point location