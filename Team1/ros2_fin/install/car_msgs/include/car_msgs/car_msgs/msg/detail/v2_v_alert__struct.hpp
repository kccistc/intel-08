// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/v2_v_alert.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_HPP_
#define CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'ts'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__car_msgs__msg__V2VAlert __attribute__((deprecated))
#else
# define DEPRECATED__car_msgs__msg__V2VAlert __declspec(deprecated)
#endif

namespace car_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct V2VAlert_
{
  using Type = V2VAlert_<ContainerAllocator>;

  explicit V2VAlert_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : ts(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ver = 0ul;
      this->src = "";
      this->seq = 0ul;
      this->type = "";
      this->severity = "";
      this->distance_m = 0.0f;
      this->road = "";
      this->lat = 0.0;
      this->lon = 0.0;
      this->suggest = "";
      this->ttl_s = 0.0f;
    }
  }

  explicit V2VAlert_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : src(_alloc),
    ts(_alloc, _init),
    type(_alloc),
    severity(_alloc),
    road(_alloc),
    suggest(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ver = 0ul;
      this->src = "";
      this->seq = 0ul;
      this->type = "";
      this->severity = "";
      this->distance_m = 0.0f;
      this->road = "";
      this->lat = 0.0;
      this->lon = 0.0;
      this->suggest = "";
      this->ttl_s = 0.0f;
    }
  }

  // field types and members
  using _ver_type =
    uint32_t;
  _ver_type ver;
  using _src_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _src_type src;
  using _seq_type =
    uint32_t;
  _seq_type seq;
  using _ts_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _ts_type ts;
  using _type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _type_type type;
  using _severity_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _severity_type severity;
  using _distance_m_type =
    float;
  _distance_m_type distance_m;
  using _road_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _road_type road;
  using _lat_type =
    double;
  _lat_type lat;
  using _lon_type =
    double;
  _lon_type lon;
  using _suggest_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _suggest_type suggest;
  using _ttl_s_type =
    float;
  _ttl_s_type ttl_s;

  // setters for named parameter idiom
  Type & set__ver(
    const uint32_t & _arg)
  {
    this->ver = _arg;
    return *this;
  }
  Type & set__src(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->src = _arg;
    return *this;
  }
  Type & set__seq(
    const uint32_t & _arg)
  {
    this->seq = _arg;
    return *this;
  }
  Type & set__ts(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->ts = _arg;
    return *this;
  }
  Type & set__type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__severity(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->severity = _arg;
    return *this;
  }
  Type & set__distance_m(
    const float & _arg)
  {
    this->distance_m = _arg;
    return *this;
  }
  Type & set__road(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->road = _arg;
    return *this;
  }
  Type & set__lat(
    const double & _arg)
  {
    this->lat = _arg;
    return *this;
  }
  Type & set__lon(
    const double & _arg)
  {
    this->lon = _arg;
    return *this;
  }
  Type & set__suggest(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->suggest = _arg;
    return *this;
  }
  Type & set__ttl_s(
    const float & _arg)
  {
    this->ttl_s = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    car_msgs::msg::V2VAlert_<ContainerAllocator> *;
  using ConstRawPtr =
    const car_msgs::msg::V2VAlert_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      car_msgs::msg::V2VAlert_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      car_msgs::msg::V2VAlert_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__car_msgs__msg__V2VAlert
    std::shared_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__car_msgs__msg__V2VAlert
    std::shared_ptr<car_msgs::msg::V2VAlert_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const V2VAlert_ & other) const
  {
    if (this->ver != other.ver) {
      return false;
    }
    if (this->src != other.src) {
      return false;
    }
    if (this->seq != other.seq) {
      return false;
    }
    if (this->ts != other.ts) {
      return false;
    }
    if (this->type != other.type) {
      return false;
    }
    if (this->severity != other.severity) {
      return false;
    }
    if (this->distance_m != other.distance_m) {
      return false;
    }
    if (this->road != other.road) {
      return false;
    }
    if (this->lat != other.lat) {
      return false;
    }
    if (this->lon != other.lon) {
      return false;
    }
    if (this->suggest != other.suggest) {
      return false;
    }
    if (this->ttl_s != other.ttl_s) {
      return false;
    }
    return true;
  }
  bool operator!=(const V2VAlert_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct V2VAlert_

// alias to use template instance with default allocator
using V2VAlert =
  car_msgs::msg::V2VAlert_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_HPP_
