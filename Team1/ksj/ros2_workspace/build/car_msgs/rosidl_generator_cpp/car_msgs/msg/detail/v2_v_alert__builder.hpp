// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/v2_v_alert.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__V2_V_ALERT__BUILDER_HPP_
#define CAR_MSGS__MSG__DETAIL__V2_V_ALERT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "car_msgs/msg/detail/v2_v_alert__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace car_msgs
{

namespace msg
{

namespace builder
{

class Init_V2VAlert_distance
{
public:
  explicit Init_V2VAlert_distance(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  ::car_msgs::msg::V2VAlert distance(::car_msgs::msg::V2VAlert::_distance_type arg)
  {
    msg_.distance = std::move(arg);
    return std::move(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_vehicle_id
{
public:
  explicit Init_V2VAlert_vehicle_id(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_distance vehicle_id(::car_msgs::msg::V2VAlert::_vehicle_id_type arg)
  {
    msg_.vehicle_id = std::move(arg);
    return Init_V2VAlert_distance(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_msg_type
{
public:
  Init_V2VAlert_msg_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_V2VAlert_vehicle_id msg_type(::car_msgs::msg::V2VAlert::_msg_type_type arg)
  {
    msg_.msg_type = std::move(arg);
    return Init_V2VAlert_vehicle_id(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::car_msgs::msg::V2VAlert>()
{
  return car_msgs::msg::builder::Init_V2VAlert_msg_type();
}

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__BUILDER_HPP_
