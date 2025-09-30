// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from car_msgs:msg/LaneInfo.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/lane_info.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__LANE_INFO__BUILDER_HPP_
#define CAR_MSGS__MSG__DETAIL__LANE_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "car_msgs/msg/detail/lane_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace car_msgs
{

namespace msg
{

namespace builder
{

class Init_LaneInfo_offset
{
public:
  explicit Init_LaneInfo_offset(::car_msgs::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  ::car_msgs::msg::LaneInfo offset(::car_msgs::msg::LaneInfo::_offset_type arg)
  {
    msg_.offset = std::move(arg);
    return std::move(msg_);
  }

private:
  ::car_msgs::msg::LaneInfo msg_;
};

class Init_LaneInfo_curvature
{
public:
  explicit Init_LaneInfo_curvature(::car_msgs::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  Init_LaneInfo_offset curvature(::car_msgs::msg::LaneInfo::_curvature_type arg)
  {
    msg_.curvature = std::move(arg);
    return Init_LaneInfo_offset(msg_);
  }

private:
  ::car_msgs::msg::LaneInfo msg_;
};

class Init_LaneInfo_is_detected
{
public:
  Init_LaneInfo_is_detected()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LaneInfo_curvature is_detected(::car_msgs::msg::LaneInfo::_is_detected_type arg)
  {
    msg_.is_detected = std::move(arg);
    return Init_LaneInfo_curvature(msg_);
  }

private:
  ::car_msgs::msg::LaneInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::car_msgs::msg::LaneInfo>()
{
  return car_msgs::msg::builder::Init_LaneInfo_is_detected();
}

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__LANE_INFO__BUILDER_HPP_
