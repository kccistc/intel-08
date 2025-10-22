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

class Init_V2VAlert_ttl_s
{
public:
  explicit Init_V2VAlert_ttl_s(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  ::car_msgs::msg::V2VAlert ttl_s(::car_msgs::msg::V2VAlert::_ttl_s_type arg)
  {
    msg_.ttl_s = std::move(arg);
    return std::move(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_suggest
{
public:
  explicit Init_V2VAlert_suggest(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_ttl_s suggest(::car_msgs::msg::V2VAlert::_suggest_type arg)
  {
    msg_.suggest = std::move(arg);
    return Init_V2VAlert_ttl_s(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_lon
{
public:
  explicit Init_V2VAlert_lon(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_suggest lon(::car_msgs::msg::V2VAlert::_lon_type arg)
  {
    msg_.lon = std::move(arg);
    return Init_V2VAlert_suggest(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_lat
{
public:
  explicit Init_V2VAlert_lat(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_lon lat(::car_msgs::msg::V2VAlert::_lat_type arg)
  {
    msg_.lat = std::move(arg);
    return Init_V2VAlert_lon(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_road
{
public:
  explicit Init_V2VAlert_road(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_lat road(::car_msgs::msg::V2VAlert::_road_type arg)
  {
    msg_.road = std::move(arg);
    return Init_V2VAlert_lat(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_distance_m
{
public:
  explicit Init_V2VAlert_distance_m(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_road distance_m(::car_msgs::msg::V2VAlert::_distance_m_type arg)
  {
    msg_.distance_m = std::move(arg);
    return Init_V2VAlert_road(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_severity
{
public:
  explicit Init_V2VAlert_severity(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_distance_m severity(::car_msgs::msg::V2VAlert::_severity_type arg)
  {
    msg_.severity = std::move(arg);
    return Init_V2VAlert_distance_m(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_type
{
public:
  explicit Init_V2VAlert_type(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_severity type(::car_msgs::msg::V2VAlert::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_V2VAlert_severity(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_ts
{
public:
  explicit Init_V2VAlert_ts(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_type ts(::car_msgs::msg::V2VAlert::_ts_type arg)
  {
    msg_.ts = std::move(arg);
    return Init_V2VAlert_type(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_seq
{
public:
  explicit Init_V2VAlert_seq(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_ts seq(::car_msgs::msg::V2VAlert::_seq_type arg)
  {
    msg_.seq = std::move(arg);
    return Init_V2VAlert_ts(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_src
{
public:
  explicit Init_V2VAlert_src(::car_msgs::msg::V2VAlert & msg)
  : msg_(msg)
  {}
  Init_V2VAlert_seq src(::car_msgs::msg::V2VAlert::_src_type arg)
  {
    msg_.src = std::move(arg);
    return Init_V2VAlert_seq(msg_);
  }

private:
  ::car_msgs::msg::V2VAlert msg_;
};

class Init_V2VAlert_ver
{
public:
  Init_V2VAlert_ver()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_V2VAlert_src ver(::car_msgs::msg::V2VAlert::_ver_type arg)
  {
    msg_.ver = std::move(arg);
    return Init_V2VAlert_src(msg_);
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
  return car_msgs::msg::builder::Init_V2VAlert_ver();
}

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__BUILDER_HPP_
