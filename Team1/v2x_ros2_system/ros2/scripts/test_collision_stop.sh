#!/usr/bin/env bash
set -eo pipefail
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-10}
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash

# 1) 기대값 검사기 먼저 실행 (백그라운드)
python3 /ws/scripts/test_expect_cmd.py --expected-linear-x 0.0 --expected-angular-z 0.0 --timeout 10.0 > /tmp/expect_collision.log 2>&1 &
EXP_PID=$!
sleep 0.5

# 2) 그 다음 퍼블리시(여러 번)
NOW=$(date +%s)
for i in $(seq 1 10); do
  ros2 topic pub -1 /v2x/alert_struct car_msgs/msg/V2VAlert "{ver: 1, src: sim, seq: $((100 + i)), ts: {sec: ${NOW}, nanosec: 0}, type: collision, severity: high, distance_m: 5.0, road: A1, lat: 0.0, lon: 0.0, suggest: \"\", ttl_s: 10.0}" >/dev/null
done

# 3) 검사기 종료 대기 + 결과 표시
wait $EXP_PID; RC=$?
cat /tmp/expect_collision.log || true
echo ">>> test_collision_stop EXIT=$RC"
exit $RC
