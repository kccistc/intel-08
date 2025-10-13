// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "car_msgs/msg/detail/emergency_event__functions.h"
#include "car_msgs/msg/detail/emergency_event__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace car_msgs
{

namespace msg
{

namespace rosidl_typesupport_cpp
{

typedef struct _EmergencyEvent_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _EmergencyEvent_type_support_ids_t;

static const _EmergencyEvent_type_support_ids_t _EmergencyEvent_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _EmergencyEvent_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _EmergencyEvent_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _EmergencyEvent_type_support_symbol_names_t _EmergencyEvent_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, car_msgs, msg, EmergencyEvent)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, car_msgs, msg, EmergencyEvent)),
  }
};

typedef struct _EmergencyEvent_type_support_data_t
{
  void * data[2];
} _EmergencyEvent_type_support_data_t;

static _EmergencyEvent_type_support_data_t _EmergencyEvent_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _EmergencyEvent_message_typesupport_map = {
  2,
  "car_msgs",
  &_EmergencyEvent_message_typesupport_ids.typesupport_identifier[0],
  &_EmergencyEvent_message_typesupport_symbol_names.symbol_name[0],
  &_EmergencyEvent_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t EmergencyEvent_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_EmergencyEvent_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &car_msgs__msg__EmergencyEvent__get_type_hash,
  &car_msgs__msg__EmergencyEvent__get_type_description,
  &car_msgs__msg__EmergencyEvent__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace msg

}  // namespace car_msgs

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<car_msgs::msg::EmergencyEvent>()
{
  return &::car_msgs::msg::rosidl_typesupport_cpp::EmergencyEvent_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, car_msgs, msg, EmergencyEvent)() {
  return get_message_type_support_handle<car_msgs::msg::EmergencyEvent>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp
