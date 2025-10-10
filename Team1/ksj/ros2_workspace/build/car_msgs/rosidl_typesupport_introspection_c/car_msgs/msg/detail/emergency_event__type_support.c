// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "car_msgs/msg/detail/emergency_event__rosidl_typesupport_introspection_c.h"
#include "car_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "car_msgs/msg/detail/emergency_event__functions.h"
#include "car_msgs/msg/detail/emergency_event__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `vehicle_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  car_msgs__msg__EmergencyEvent__init(message_memory);
}

void car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_fini_function(void * message_memory)
{
  car_msgs__msg__EmergencyEvent__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_member_array[5] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs__msg__EmergencyEvent, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "msg_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs__msg__EmergencyEvent, msg_type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vehicle_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs__msg__EmergencyEvent, vehicle_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs__msg__EmergencyEvent, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "confidence_score",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(car_msgs__msg__EmergencyEvent, confidence_score),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_members = {
  "car_msgs__msg",  // message namespace
  "EmergencyEvent",  // message name
  5,  // number of fields
  sizeof(car_msgs__msg__EmergencyEvent),
  false,  // has_any_key_member_
  car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_member_array,  // message members
  car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_init_function,  // function to initialize message memory (memory has to be allocated)
  car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_type_support_handle = {
  0,
  &car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_members,
  get_message_typesupport_handle_function,
  &car_msgs__msg__EmergencyEvent__get_type_hash,
  &car_msgs__msg__EmergencyEvent__get_type_description,
  &car_msgs__msg__EmergencyEvent__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_car_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, car_msgs, msg, EmergencyEvent)() {
  car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_type_support_handle.typesupport_identifier) {
    car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &car_msgs__msg__EmergencyEvent__rosidl_typesupport_introspection_c__EmergencyEvent_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
