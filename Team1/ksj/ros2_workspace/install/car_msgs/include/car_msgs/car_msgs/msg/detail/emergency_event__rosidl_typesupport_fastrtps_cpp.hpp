// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

#ifndef CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include <cstddef>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "car_msgs/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "car_msgs/msg/detail/emergency_event__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace car_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
cdr_serialize(
  const car_msgs::msg::EmergencyEvent & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  car_msgs::msg::EmergencyEvent & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
get_serialized_size(
  const car_msgs::msg::EmergencyEvent & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
max_serialized_size_EmergencyEvent(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
cdr_serialize_key(
  const car_msgs::msg::EmergencyEvent & ros_message,
  eprosima::fastcdr::Cdr &);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
get_serialized_size_key(
  const car_msgs::msg::EmergencyEvent & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
max_serialized_size_key_EmergencyEvent(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace car_msgs

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_car_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, car_msgs, msg, EmergencyEvent)();

#ifdef __cplusplus
}
#endif

#endif  // CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
