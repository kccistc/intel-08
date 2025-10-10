// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

#include "car_msgs/msg/detail/emergency_event__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_car_msgs
const rosidl_type_hash_t *
car_msgs__msg__EmergencyEvent__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xcf, 0xb3, 0x10, 0xf9, 0x5f, 0x77, 0x11, 0x23,
      0xdc, 0x11, 0xa3, 0x83, 0x57, 0x0f, 0xcd, 0x46,
      0xe0, 0x7c, 0x81, 0xe1, 0xaa, 0x36, 0x86, 0x74,
      0xfa, 0xa7, 0xfd, 0x73, 0xc0, 0xf7, 0xaf, 0x01,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "std_msgs/msg/detail/header__functions.h"
#include "geometry_msgs/msg/detail/point__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t geometry_msgs__msg__Point__EXPECTED_HASH = {1, {
    0x69, 0x63, 0x08, 0x48, 0x42, 0xa9, 0xb0, 0x44,
    0x94, 0xd6, 0xb2, 0x94, 0x1d, 0x11, 0x44, 0x47,
    0x08, 0xd8, 0x92, 0xda, 0x2f, 0x4b, 0x09, 0x84,
    0x3b, 0x9c, 0x43, 0xf4, 0x2a, 0x7f, 0x68, 0x81,
  }};
static const rosidl_type_hash_t std_msgs__msg__Header__EXPECTED_HASH = {1, {
    0xf4, 0x9f, 0xb3, 0xae, 0x2c, 0xf0, 0x70, 0xf7,
    0x93, 0x64, 0x5f, 0xf7, 0x49, 0x68, 0x3a, 0xc6,
    0xb0, 0x62, 0x03, 0xe4, 0x1c, 0x89, 0x1e, 0x17,
    0x70, 0x1b, 0x1c, 0xb5, 0x97, 0xce, 0x6a, 0x01,
  }};
#endif

static char car_msgs__msg__EmergencyEvent__TYPE_NAME[] = "car_msgs/msg/EmergencyEvent";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char geometry_msgs__msg__Point__TYPE_NAME[] = "geometry_msgs/msg/Point";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char car_msgs__msg__EmergencyEvent__FIELD_NAME__header[] = "header";
static char car_msgs__msg__EmergencyEvent__FIELD_NAME__msg_type[] = "msg_type";
static char car_msgs__msg__EmergencyEvent__FIELD_NAME__vehicle_id[] = "vehicle_id";
static char car_msgs__msg__EmergencyEvent__FIELD_NAME__position[] = "position";
static char car_msgs__msg__EmergencyEvent__FIELD_NAME__confidence_score[] = "confidence_score";

static rosidl_runtime_c__type_description__Field car_msgs__msg__EmergencyEvent__FIELDS[] = {
  {
    {car_msgs__msg__EmergencyEvent__FIELD_NAME__header, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__EmergencyEvent__FIELD_NAME__msg_type, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__EmergencyEvent__FIELD_NAME__vehicle_id, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__EmergencyEvent__FIELD_NAME__position, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__EmergencyEvent__FIELD_NAME__confidence_score, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription car_msgs__msg__EmergencyEvent__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
car_msgs__msg__EmergencyEvent__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {car_msgs__msg__EmergencyEvent__TYPE_NAME, 27, 27},
      {car_msgs__msg__EmergencyEvent__FIELDS, 5, 5},
    },
    {car_msgs__msg__EmergencyEvent__REFERENCED_TYPE_DESCRIPTIONS, 3, 3},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Point__EXPECTED_HASH, geometry_msgs__msg__Point__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = geometry_msgs__msg__Point__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "uint8 MSG_TYPE_EMERGENCY_BRAKE = 0\n"
  "uint8 MSG_TYPE_OBSTACLE_AHEAD = 1\n"
  "uint8 MSG_TYPE_RECKLESS_DRIVING = 2\n"
  "uint8 MSG_TYPE_UNKNOWN = 255\n"
  "\n"
  "# \\xeb\\xa9\\x94\\xec\\x8b\\x9c\\xec\\xa7\\x80\\xec\\x9d\\x98 \\xed\\x83\\x80\\xec\\x9e\\x84\\xec\\x8a\\xa4\\xed\\x83\\xac\\xed\\x94\\x84\\xec\\x99\\x80 \\xed\\x94\\x84\\xeb\\xa0\\x88\\xec\\x9e\\x84 ID\n"
  "std_msgs/Header header\n"
  "\n"
  "# \\xeb\\xa9\\x94\\xec\\x8b\\x9c\\xec\\xa7\\x80 \\xec\\xa2\\x85\\xeb\\xa5\\x98 (\\xec\\x9c\\x84\\xec\\x9d\\x98 \\xed\\x83\\x80\\xec\\x9e\\x85 \\xec\\x83\\x81\\xec\\x88\\x98 \\xec\\xa4\\x91 \\xed\\x95\\x98\\xeb\\x82\\x98)\n"
  "uint8 msg_type\n"
  "\n"
  "# \\xec\\xa0\\x95\\xeb\\xb3\\xb4\\xeb\\xa5\\xbc \\xeb\\xb3\\xb4\\xeb\\x82\\xb8 \\xec\\xb0\\xa8\\xeb\\x9f\\x89\\xec\\x9d\\x98 ID\n"
  "string vehicle_id\n"
  "\n"
  "# \\xec\\x9d\\xb4\\xeb\\xb2\\xa4\\xed\\x8a\\xb8 \\xeb\\xb0\\x9c\\xec\\x83\\x9d \\xec\\xa7\\x80\\xec\\xa0\\x90 \\xec\\xa2\\x8c\\xed\\x91\\x9c (geometry_msgs/Point \\xed\\x83\\x80\\xec\\x9e\\x85 \\xec\\x82\\xac\\xec\\x9a\\xa9)\n"
  "geometry_msgs/Point position\n"
  "\n"
  "# \\xeb\\xa9\\x94\\xec\\x8b\\x9c\\xec\\xa7\\x80\\xec\\x9d\\x98 \\xec\\x8b\\xa0\\xeb\\xa2\\xb0\\xeb\\x8f\\x84 (0.0 ~ 1.0)\n"
  "float32 confidence_score";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
car_msgs__msg__EmergencyEvent__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {car_msgs__msg__EmergencyEvent__TYPE_NAME, 27, 27},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 376, 376},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
car_msgs__msg__EmergencyEvent__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[4];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 4, 4};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *car_msgs__msg__EmergencyEvent__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *geometry_msgs__msg__Point__get_individual_type_description_source(NULL);
    sources[3] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
