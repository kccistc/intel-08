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
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->msg_type = 0;
      this->vehicle_id = "";
      this->distance = 0.0;
    }
  }

  explicit V2VAlert_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : vehicle_id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->msg_type = 0;
      this->vehicle_id = "";
      this->distance = 0.0;
    }
  }

  // field types and members
  using _msg_type_type =
    uint8_t;
  _msg_type_type msg_type;
  using _vehicle_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _vehicle_id_type vehicle_id;
  using _distance_type =
    double;
  _distance_type distance;

  // setters for named parameter idiom
  Type & set__msg_type(
    const uint8_t & _arg)
  {
    this->msg_type = _arg;
    return *this;
  }
  Type & set__vehicle_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->vehicle_id = _arg;
    return *this;
  }
  Type & set__distance(
    const double & _arg)
  {
    this->distance = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t MSG_TYPE_EMERGENCY_BRAKE =
    0u;
  static constexpr uint8_t MSG_TYPE_OBSTACLE_AHEAD =
    1u;

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
    if (this->msg_type != other.msg_type) {
      return false;
    }
    if (this->vehicle_id != other.vehicle_id) {
      return false;
    }
    if (this->distance != other.distance) {
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
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t V2VAlert_<ContainerAllocator>::MSG_TYPE_EMERGENCY_BRAKE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t V2VAlert_<ContainerAllocator>::MSG_TYPE_OBSTACLE_AHEAD;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__V2_V_ALERT__STRUCT_HPP_
