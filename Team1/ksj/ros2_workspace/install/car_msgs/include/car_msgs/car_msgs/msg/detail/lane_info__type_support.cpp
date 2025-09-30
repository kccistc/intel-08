// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from car_msgs:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "car_msgs/msg/detail/lane_info__functions.h"
#include "car_msgs/msg/detail/lane_info__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace car_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void LaneInfo_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) car_msgs::msg::LaneInfo(_init);
}

void LaneInfo_fini_function(void * message_memory)
{
  auto typed_message = static_cast<car_msgs::msg::LaneInfo *>(message_memory);
  typed_message->~LaneInfo();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember LaneInfo_message_member_array[3] = {
  {
    "is_detected",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs::msg::LaneInfo, is_detected),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "curvature",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs::msg::LaneInfo, curvature),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "offset",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs::msg::LaneInfo, offset),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers LaneInfo_message_members = {
  "car_msgs::msg",  // message namespace
  "LaneInfo",  // message name
  3,  // number of fields
  sizeof(car_msgs::msg::LaneInfo),
  false,  // has_any_key_member_
  LaneInfo_message_member_array,  // message members
  LaneInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  LaneInfo_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t LaneInfo_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &LaneInfo_message_members,
  get_message_typesupport_handle_function,
  &car_msgs__msg__LaneInfo__get_type_hash,
  &car_msgs__msg__LaneInfo__get_type_description,
  &car_msgs__msg__LaneInfo__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace car_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<car_msgs::msg::LaneInfo>()
{
  return &::car_msgs::msg::rosidl_typesupport_introspection_cpp::LaneInfo_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, car_msgs, msg, LaneInfo)() {
  return &::car_msgs::msg::rosidl_typesupport_introspection_cpp::LaneInfo_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
