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

/// Constant 'MSG_TYPE_EMERGENCY_BRAKE'.
enum
{
  car_msgs__msg__V2VAlert__MSG_TYPE_EMERGENCY_BRAKE = 0
};

/// Constant 'MSG_TYPE_OBSTACLE_AHEAD'.
enum
{
  car_msgs__msg__V2VAlert__MSG_TYPE_OBSTACLE_AHEAD = 1
};

// Include directives for member types
// Member 'vehicle_id'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/V2VAlert in the package car_msgs.
typedef struct car_msgs__msg__V2VAlert
{
  /// 메시지 종류 (위의 타입 중 하나)
  uint8_t msg_type;
  /// 정보를 보낸 차량의 ID
  rosidl_runtime_c__String vehicle_id;
  /// 이벤트 발생 지점까지의 거리 (m)
  double distance;
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
