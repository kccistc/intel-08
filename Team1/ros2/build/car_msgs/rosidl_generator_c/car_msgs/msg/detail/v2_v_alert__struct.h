// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/v2_v_alert.h"


#ifndef CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_H_
#define CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'src'
// Member 'type'
// Member 'severity'
// Member 'road'
// Member 'suggest'
#include "rosidl_runtime_c/string.h"
// Member 'ts'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/V2VAlert in the package car_msgs.
typedef struct car_msgs__msg__V2VAlert
{
  uint32_t ver;
  rosidl_runtime_c__String src;
  uint32_t seq;
  /// 수신/발행 시각(또는 원본 ts 변환)
  builtin_interfaces__msg__Time ts;
  /// "collision", "fire", ...
  rosidl_runtime_c__String type;
  /// "low"|"medium"|"high"
  rosidl_runtime_c__String severity;
  float distance_m;
  rosidl_runtime_c__String road;
  double lat;
  double lon;
  /// "slow_down"|"stop"|"reroute"|...
  rosidl_runtime_c__String suggest;
  float ttl_s;
} car_msgs__msg__V2VAlert;

// Struct for a sequence of car_msgs__msg__V2VAlert.
typedef struct car_msgs__msg__V2VAlert__Sequence
{
  car_msgs__msg__V2VAlert * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} car_msgs__msg__V2VAlert__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_H_
