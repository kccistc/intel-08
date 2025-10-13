// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from car_msgs:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#include "car_msgs/msg/detail/lane_info__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_car_msgs
const rosidl_type_hash_t *
car_msgs__msg__LaneInfo__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x63, 0xa5, 0x5f, 0xdd, 0xd0, 0x95, 0x84, 0x20,
      0x6c, 0x5d, 0x2d, 0xde, 0xad, 0x78, 0xc7, 0x85,
      0x17, 0x5a, 0x16, 0x13, 0x10, 0x08, 0xe8, 0xb2,
      0xe4, 0x0f, 0x3f, 0xc4, 0x46, 0xde, 0x70, 0xcd,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char car_msgs__msg__LaneInfo__TYPE_NAME[] = "car_msgs/msg/LaneInfo";

// Define type names, field names, and default values
static char car_msgs__msg__LaneInfo__FIELD_NAME__is_detected[] = "is_detected";
static char car_msgs__msg__LaneInfo__FIELD_NAME__curvature[] = "curvature";
static char car_msgs__msg__LaneInfo__FIELD_NAME__offset[] = "offset";

static rosidl_runtime_c__type_description__Field car_msgs__msg__LaneInfo__FIELDS[] = {
  {
    {car_msgs__msg__LaneInfo__FIELD_NAME__is_detected, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__LaneInfo__FIELD_NAME__curvature, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__LaneInfo__FIELD_NAME__offset, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
car_msgs__msg__LaneInfo__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {car_msgs__msg__LaneInfo__TYPE_NAME, 21, 21},
      {car_msgs__msg__LaneInfo__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "\n"
  "bool is_detected      # \\xec\\xb0\\xa8\\xec\\x84\\xa0 \\xea\\xb0\\x90\\xec\\xa7\\x80 \\xec\\x97\\xac\\xeb\\xb6\\x80\n"
  "float32 curvature     # \\xec\\xb0\\xa8\\xec\\x84\\xa0\\xec\\x9d\\x98 \\xea\\xb3\\xa1\\xeb\\xa5\\xa0 (0\\xec\\x97\\x90 \\xea\\xb0\\x80\\xea\\xb9\\x8c\\xec\\x9a\\xb8\\xec\\x88\\x98\\xeb\\xa1\\x9d \\xec\\xa7\\x81\\xec\\x84\\xa0)\n"
  "float32 offset        # \\xec\\xb0\\xa8\\xeb\\x9f\\x89 \\xec\\xa4\\x91\\xec\\x8b\\xac\\xec\\x9c\\xbc\\xeb\\xa1\\x9c\\xeb\\xb6\\x80\\xed\\x84\\xb0 \\xec\\xb0\\xa8\\xec\\x84\\xa0 \\xec\\xa4\\x91\\xec\\x95\\x99\\xea\\xb9\\x8c\\xec\\xa7\\x80\\xec\\x9d\\x98 \\xea\\xb1\\xb0\\xeb\\xa6\\xac (m)";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
car_msgs__msg__LaneInfo__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {car_msgs__msg__LaneInfo__TYPE_NAME, 21, 21},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 128, 128},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
car_msgs__msg__LaneInfo__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *car_msgs__msg__LaneInfo__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
