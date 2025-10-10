// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/emergency_event.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__BUILDER_HPP_
#define CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "car_msgs/msg/detail/emergency_event__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace car_msgs
{

namespace msg
{

namespace builder
{

class Init_EmergencyEvent_confidence_score
{
public:
  explicit Init_EmergencyEvent_confidence_score(::car_msgs::msg::EmergencyEvent & msg)
  : msg_(msg)
  {}
  ::car_msgs::msg::EmergencyEvent confidence_score(::car_msgs::msg::EmergencyEvent::_confidence_score_type arg)
  {
    msg_.confidence_score = std::move(arg);
    return std::move(msg_);
  }

private:
  ::car_msgs::msg::EmergencyEvent msg_;
};

class Init_EmergencyEvent_position
{
public:
  explicit Init_EmergencyEvent_position(::car_msgs::msg::EmergencyEvent & msg)
  : msg_(msg)
  {}
  Init_EmergencyEvent_confidence_score position(::car_msgs::msg::EmergencyEvent::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_EmergencyEvent_confidence_score(msg_);
  }

private:
  ::car_msgs::msg::EmergencyEvent msg_;
};

class Init_EmergencyEvent_vehicle_id
{
public:
  explicit Init_EmergencyEvent_vehicle_id(::car_msgs::msg::EmergencyEvent & msg)
  : msg_(msg)
  {}
  Init_EmergencyEvent_position vehicle_id(::car_msgs::msg::EmergencyEvent::_vehicle_id_type arg)
  {
    msg_.vehicle_id = std::move(arg);
    return Init_EmergencyEvent_position(msg_);
  }

private:
  ::car_msgs::msg::EmergencyEvent msg_;
};

class Init_EmergencyEvent_msg_type
{
public:
  explicit Init_EmergencyEvent_msg_type(::car_msgs::msg::EmergencyEvent & msg)
  : msg_(msg)
  {}
  Init_EmergencyEvent_vehicle_id msg_type(::car_msgs::msg::EmergencyEvent::_msg_type_type arg)
  {
    msg_.msg_type = std::move(arg);
    return Init_EmergencyEvent_vehicle_id(msg_);
  }

private:
  ::car_msgs::msg::EmergencyEvent msg_;
};

class Init_EmergencyEvent_header
{
public:
  Init_EmergencyEvent_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EmergencyEvent_msg_type header(::car_msgs::msg::EmergencyEvent::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_EmergencyEvent_msg_type(msg_);
  }

private:
  ::car_msgs::msg::EmergencyEvent msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::car_msgs::msg::EmergencyEvent>()
{
  return car_msgs::msg::builder::Init_EmergencyEvent_header();
}

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__BUILDER_HPP_
