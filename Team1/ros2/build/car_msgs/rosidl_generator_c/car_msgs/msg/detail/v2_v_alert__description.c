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
      0xeb, 0x1d, 0x63, 0xc3, 0x12, 0x86, 0x1d, 0x10,
      0x97, 0x54, 0x05, 0x91, 0x73, 0xd3, 0x18, 0xb0,
      0x28, 0xbe, 0xdb, 0x6b, 0xe3, 0xce, 0x29, 0xd7,
      0x6d, 0x5d, 0xec, 0x85, 0xc0, 0xdc, 0x5b, 0xc4,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
#endif

static char car_msgs__msg__V2VAlert__TYPE_NAME[] = "car_msgs/msg/V2VAlert";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";

// Define type names, field names, and default values
static char car_msgs__msg__V2VAlert__FIELD_NAME__ver[] = "ver";
static char car_msgs__msg__V2VAlert__FIELD_NAME__src[] = "src";
static char car_msgs__msg__V2VAlert__FIELD_NAME__seq[] = "seq";
static char car_msgs__msg__V2VAlert__FIELD_NAME__ts[] = "ts";
static char car_msgs__msg__V2VAlert__FIELD_NAME__type[] = "type";
static char car_msgs__msg__V2VAlert__FIELD_NAME__severity[] = "severity";
static char car_msgs__msg__V2VAlert__FIELD_NAME__distance_m[] = "distance_m";
static char car_msgs__msg__V2VAlert__FIELD_NAME__road[] = "road";
static char car_msgs__msg__V2VAlert__FIELD_NAME__lat[] = "lat";
static char car_msgs__msg__V2VAlert__FIELD_NAME__lon[] = "lon";
static char car_msgs__msg__V2VAlert__FIELD_NAME__suggest[] = "suggest";
static char car_msgs__msg__V2VAlert__FIELD_NAME__ttl_s[] = "ttl_s";

static rosidl_runtime_c__type_description__Field car_msgs__msg__V2VAlert__FIELDS[] = {
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__ver, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__src, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__seq, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__ts, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__type, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__severity, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__distance_m, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__road, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__lat, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__lon, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__suggest, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {car_msgs__msg__V2VAlert__FIELD_NAME__ttl_s, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription car_msgs__msg__V2VAlert__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
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
      {car_msgs__msg__V2VAlert__FIELDS, 12, 12},
    },
    {car_msgs__msg__V2VAlert__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "uint32 ver\n"
  "string src\n"
  "uint32 seq\n"
  "builtin_interfaces/Time ts  # \\xec\\x88\\x98\\xec\\x8b\\xa0/\\xeb\\xb0\\x9c\\xed\\x96\\x89 \\xec\\x8b\\x9c\\xea\\xb0\\x81(\\xeb\\x98\\x90\\xeb\\x8a\\x94 \\xec\\x9b\\x90\\xeb\\xb3\\xb8 ts \\xeb\\xb3\\x80\\xed\\x99\\x98)\n"
  "\n"
  "string type          # \"collision\", \"fire\", ...\n"
  "string severity      # \"low\"|\"medium\"|\"high\"\n"
  "float32 distance_m\n"
  "string road\n"
  "float64 lat\n"
  "float64 lon\n"
  "\n"
  "string suggest       # \"slow_down\"|\"stop\"|\"reroute\"|...\n"
  "\n"
  "float32 ttl_s";

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
    {toplevel_type_raw_source, 306, 306},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
car_msgs__msg__V2VAlert__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *car_msgs__msg__V2VAlert__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
