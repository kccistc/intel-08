// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice
#ifndef CAR_MSGS__MSG__DETAIL__V2_V_ALERT__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define CAR_MSGS__MSG__DETAIL__V2_V_ALERT__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "car_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "car_msgs/msg/detail/v2_v_alert__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
bool cdr_serialize_car_msgs__msg__V2VAlert(
  const car_msgs__msg__V2VAlert * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
bool cdr_deserialize_car_msgs__msg__V2VAlert(
  eprosima::fastcdr::Cdr &,
  car_msgs__msg__V2VAlert * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
size_t get_serialized_size_car_msgs__msg__V2VAlert(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
size_t max_serialized_size_car_msgs__msg__V2VAlert(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
bool cdr_serialize_key_car_msgs__msg__V2VAlert(
  const car_msgs__msg__V2VAlert * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
size_t get_serialized_size_key_car_msgs__msg__V2VAlert(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
size_t max_serialized_size_key_car_msgs__msg__V2VAlert(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_car_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, car_msgs, msg, V2VAlert)();

#ifdef __cplusplus
}
#endif

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
