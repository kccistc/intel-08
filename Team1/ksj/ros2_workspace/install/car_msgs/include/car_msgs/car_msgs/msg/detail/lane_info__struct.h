// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from car_msgs:msg/LaneInfo.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/lane_info.h"


#ifndef CAR_MSGS__MSG__DETAIL__LANE_INFO__STRUCT_H_
#define CAR_MSGS__MSG__DETAIL__LANE_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/LaneInfo in the package car_msgs.
typedef struct car_msgs__msg__LaneInfo
{
  /// 차선 감지 여부
  bool is_detected;
  /// 차선의 곡률 (0에 가까울수록 직선)
  float curvature;
  /// 차량 중심으로부터 차선 중앙까지의 거리 (m)
  float offset;
} car_msgs__msg__LaneInfo;

// Struct for a sequence of car_msgs__msg__LaneInfo.
typedef struct car_msgs__msg__LaneInfo__Sequence
{
  car_msgs__msg__LaneInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} car_msgs__msg__LaneInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CAR_MSGS__MSG__DETAIL__LANE_INFO__STRUCT_H_
