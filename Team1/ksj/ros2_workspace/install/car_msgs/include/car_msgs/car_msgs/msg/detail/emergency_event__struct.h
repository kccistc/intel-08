// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/emergency_event.h"


#ifndef CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__STRUCT_H_
#define CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__STRUCT_H_

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
  car_msgs__msg__EmergencyEvent__MSG_TYPE_EMERGENCY_BRAKE = 0
};

/// Constant 'MSG_TYPE_OBSTACLE_AHEAD'.
enum
{
  car_msgs__msg__EmergencyEvent__MSG_TYPE_OBSTACLE_AHEAD = 1
};

/// Constant 'MSG_TYPE_RECKLESS_DRIVING'.
enum
{
  car_msgs__msg__EmergencyEvent__MSG_TYPE_RECKLESS_DRIVING = 2
};

/// Constant 'MSG_TYPE_UNKNOWN'.
enum
{
  car_msgs__msg__EmergencyEvent__MSG_TYPE_UNKNOWN = 255
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'vehicle_id'
#include "rosidl_runtime_c/string.h"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/EmergencyEvent in the package car_msgs.
typedef struct car_msgs__msg__EmergencyEvent
{
  /// 메시지의 타임스탬프와 프레임 ID
  std_msgs__msg__Header header;
  /// 메시지 종류 (위의 타입 상수 중 하나)
  uint8_t msg_type;
  /// 정보를 보낸 차량의 ID
  rosidl_runtime_c__String vehicle_id;
  /// 이벤트 발생 지점 좌표 (geometry_msgs/Point 타입 사용)
  geometry_msgs__msg__Point position;
  /// 메시지의 신뢰도 (0.0 ~ 1.0)
  float confidence_score;
} car_msgs__msg__EmergencyEvent;

// Struct for a sequence of car_msgs__msg__EmergencyEvent.
typedef struct car_msgs__msg__EmergencyEvent__Sequence
{
  car_msgs__msg__EmergencyEvent * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} car_msgs__msg__EmergencyEvent__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__STRUCT_H_
