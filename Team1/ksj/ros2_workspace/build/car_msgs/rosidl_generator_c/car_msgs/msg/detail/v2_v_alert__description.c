// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice

#include "car_msgs/msg/detail/v2_v_alert__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_car_msgs
const rosidl_type_hash_t *
car_msgs__msg__V2VAlert__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4f, 0xcf, 0x5a, 0x63, 0xe1, 0xdf, 0xb5, 0xbd,
      0xc4, 0x04, 0x6b, 0x90, 0xac, 0xa2, 0x7a, 0xa8,
      0x81, 0xa1, 0xd4, 0x89, 0x37, 0xd5, 0x20, 0x94,
      0x48, 0x54, 0x75, 0x6b, 0xd6, 0xe3, 0x7f, 0xf7,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char car_msgs__msg__V2VAlert__TYPE_NAME[] = "car_msgs/msg/V2VAlert";

// Define type names, field names, and default values
static char car_msgs__msg__V2VAlert__FIELD_NAME__msg_type[] = "msg_type";
static char car_msgs__msg__V2VAlert__FIELD_NAME__vehicle_id[] = "vehicle_id";
static char car_msgs__msg__V2VAlert__FIELD_NAME__distance[] = "distance";

static rosidl_runtime_c__type_description__Field car_msgs__msg__V2VAlert__FIELDS[] = {
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__msg_type, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__vehicle_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__distance, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
car_msgs__msg__V2VAlert__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {car_msgs__msg__V2VAlert__TYPE_NAME, 21, 21},
      {car_msgs__msg__V2VAlert__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "uint8 MSG_TYPE_EMERGENCY_BRAKE = 0\n"
  "uint8 MSG_TYPE_OBSTACLE_AHEAD = 1\n"
  "\n"
  "uint8 msg_type       # \\xeb\\xa9\\x94\\xec\\x8b\\x9c\\xec\\xa7\\x80 \\xec\\xa2\\x85\\xeb\\xa5\\x98 (\\xec\\x9c\\x84\\xec\\x9d\\x98 \\xed\\x83\\x80\\xec\\x9e\\x85 \\xec\\xa4\\x91 \\xed\\x95\\x98\\xeb\\x82\\x98)\n"
  "string vehicle_id    # \\xec\\xa0\\x95\\xeb\\xb3\\xb4\\xeb\\xa5\\xbc \\xeb\\xb3\\xb4\\xeb\\x82\\xb8 \\xec\\xb0\\xa8\\xeb\\x9f\\x89\\xec\\x9d\\x98 ID\n"
  "float64 distance     # \\xec\\x9d\\xb4\\xeb\\xb2\\xa4\\xed\\x8a\\xb8 \\xeb\\xb0\\x9c\\xec\\x83\\x9d \\xec\\xa7\\x80\\xec\\xa0\\x90\\xea\\xb9\\x8c\\xec\\xa7\\x80\\xec\\x9d\\x98 \\xea\\xb1\\xb0\\xeb\\xa6\\xac (m)";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
car_msgs__msg__V2VAlert__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {car_msgs__msg__V2VAlert__TYPE_NAME, 21, 21},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 192, 192},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
car_msgs__msg__V2VAlert__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *car_msgs__msg__V2VAlert__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
