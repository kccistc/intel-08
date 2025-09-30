// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from car_msgs:msg/LaneInfo.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/lane_info.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__LANE_INFO__TRAITS_HPP_
#define CAR_MSGS__MSG__DETAIL__LANE_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "car_msgs/msg/detail/lane_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace car_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const LaneInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: is_detected
  {
    out << "is_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.is_detected, out);
    out << ", ";
  }

  // member: curvature
  {
    out << "curvature: ";
    rosidl_generator_traits::value_to_yaml(msg.curvature, out);
    out << ", ";
  }

  // member: offset
  {
    out << "offset: ";
    rosidl_generator_traits::value_to_yaml(msg.offset, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const LaneInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: is_detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.is_detected, out);
    out << "\n";
  }

  // member: curvature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "curvature: ";
    rosidl_generator_traits::value_to_yaml(msg.curvature, out);
    out << "\n";
  }

  // member: offset
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "offset: ";
    rosidl_generator_traits::value_to_yaml(msg.offset, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const LaneInfo & msg, bool use_flow_style = false)
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
  const car_msgs::msg::LaneInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  car_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use car_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const car_msgs::msg::LaneInfo & msg)
{
  return car_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<car_msgs::msg::LaneInfo>()
{
  return "car_msgs::msg::LaneInfo";
}

template<>
inline const char * name<car_msgs::msg::LaneInfo>()
{
  return "car_msgs/msg/LaneInfo";
}

template<>
struct has_fixed_size<car_msgs::msg::LaneInfo>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<car_msgs::msg::LaneInfo>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<car_msgs::msg::LaneInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CAR_MSGS__MSG__DETAIL__LANE_INFO__TRAITS_HPP_
