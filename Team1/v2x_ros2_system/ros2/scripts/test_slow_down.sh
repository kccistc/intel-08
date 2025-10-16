#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash

NOW=$(date +%s)
ros2 topic pub -1 /v2x/alert_struct car_msgs/msg/V2VAlert "
ver: 1
src: 'sim'
seq: 102
ts: {sec: $NOW, nanosec: 0}
type: 'hazard'
severity: 'medium'
distance_m: 500.0
road: 'A1'
lat: 0.0
lon: 0.0
suggest: 'slow_down'
ttl_s: 10.0
" >/dev/null

exec python3 /ws/scripts/test_expect_cmd.py \
  --expected-linear-x 0.3 --expected-angular-z 0.0 --timeout 5.0
