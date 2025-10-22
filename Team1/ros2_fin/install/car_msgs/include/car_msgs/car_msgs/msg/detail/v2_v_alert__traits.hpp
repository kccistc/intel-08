// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/v2_v_alert.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__V2_V_ALERT__TRAITS_HPP_
#define CAR_MSGS__MSG__DETAIL__V2_V_ALERT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "car_msgs/msg/detail/v2_v_alert__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'ts'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace car_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const V2VAlert & msg,
  std::ostream & out)
{
  out << "{";
  // member: ver
  {
    out << "ver: ";
    rosidl_generator_traits::value_to_yaml(msg.ver, out);
    out << ", ";
  }

  // member: src
  {
    out << "src: ";
    rosidl_generator_traits::value_to_yaml(msg.src, out);
    out << ", ";
  }

  // member: seq
  {
    out << "seq: ";
    rosidl_generator_traits::value_to_yaml(msg.seq, out);
    out << ", ";
  }

  // member: ts
  {
    out << "ts: ";
    to_flow_style_yaml(msg.ts, out);
    out << ", ";
  }

  // member: type
  {
    out << "type: ";
    rosidl_generator_traits::value_to_yaml(msg.type, out);
    out << ", ";
  }

  // member: severity
  {
    out << "severity: ";
    rosidl_generator_traits::value_to_yaml(msg.severity, out);
    out << ", ";
  }

  // member: distance_m
  {
    out << "distance_m: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_m, out);
    out << ", ";
  }

  // member: road
  {
    out << "road: ";
    rosidl_generator_traits::value_to_yaml(msg.road, out);
    out << ", ";
  }

  // member: lat
  {
    out << "lat: ";
    rosidl_generator_traits::value_to_yaml(msg.lat, out);
    out << ", ";
  }

  // member: lon
  {
    out << "lon: ";
    rosidl_generator_traits::value_to_yaml(msg.lon, out);
    out << ", ";
  }

  // member: suggest
  {
    out << "suggest: ";
    rosidl_generator_traits::value_to_yaml(msg.suggest, out);
    out << ", ";
  }

  // member: ttl_s
  {
    out << "ttl_s: ";
    rosidl_generator_traits::value_to_yaml(msg.ttl_s, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const V2VAlert & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: ver
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ver: ";
    rosidl_generator_traits::value_to_yaml(msg.ver, out);
    out << "\n";
  }

  // member: src
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "src: ";
    rosidl_generator_traits::value_to_yaml(msg.src, out);
    out << "\n";
  }

  // member: seq
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "seq: ";
    rosidl_generator_traits::value_to_yaml(msg.seq, out);
    out << "\n";
  }

  // member: ts
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ts:\n";
    to_block_style_yaml(msg.ts, out, indentation + 2);
  }

  // member: type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "type: ";
    rosidl_generator_traits::value_to_yaml(msg.type, out);
    out << "\n";
  }

  // member: severity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "severity: ";
    rosidl_generator_traits::value_to_yaml(msg.severity, out);
    out << "\n";
  }

  // member: distance_m
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "distance_m: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_m, out);
    out << "\n";
  }

  // member: road
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "road: ";
    rosidl_generator_traits::value_to_yaml(msg.road, out);
    out << "\n";
  }

  // member: lat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lat: ";
    rosidl_generator_traits::value_to_yaml(msg.lat, out);
    out << "\n";
  }

  // member: lon
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lon: ";
    rosidl_generator_traits::value_to_yaml(msg.lon, out);
    out << "\n";
  }

  // member: suggest
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "suggest: ";
    rosidl_generator_traits::value_to_yaml(msg.suggest, out);
    out << "\n";
  }

  // member: ttl_s
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ttl_s: ";
    rosidl_generator_traits::value_to_yaml(msg.ttl_s, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const V2VAlert & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace car_msgs

namespace rosidl_generator_traits
{

[[deprecated("use car_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const car_msgs::msg::V2VAlert & msg,
  std::ostream & out, size_t indentation = 0)
{
  car_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use car_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const car_msgs::msg::V2VAlert & msg)
{
  return car_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<car_msgs::msg::V2VAlert>()
{
  return "car_msgs::msg::V2VAlert";
}

template<>
inline const char * name<car_msgs::msg::V2VAlert>()
{
  return "car_msgs/msg/V2VAlert";
}

template<>
struct has_fixed_size<car_msgs::msg::V2VAlert>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<car_msgs::msg::V2VAlert>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<car_msgs::msg::V2VAlert>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__TRAITS_HPP_
