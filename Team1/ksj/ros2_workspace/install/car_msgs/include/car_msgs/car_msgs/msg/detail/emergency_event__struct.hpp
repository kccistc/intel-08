// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "car_msgs/msg/emergency_event.hpp"


#ifndef CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__STRUCT_HPP_
#define CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__car_msgs__msg__EmergencyEvent __attribute__((deprecated))
#else
# define DEPRECATED__car_msgs__msg__EmergencyEvent __declspec(deprecated)
#endif

namespace car_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct EmergencyEvent_
{
  using Type = EmergencyEvent_<ContainerAllocator>;

  explicit EmergencyEvent_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->msg_type = 0;
      this->vehicle_id = "";
      this->confidence_score = 0.0f;
    }
  }

  explicit EmergencyEvent_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    vehicle_id(_alloc),
    position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->msg_type = 0;
      this->vehicle_id = "";
      this->confidence_score = 0.0f;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _msg_type_type =
    uint8_t;
  _msg_type_type msg_type;
  using _vehicle_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _vehicle_id_type vehicle_id;
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _confidence_score_type =
    float;
  _confidence_score_type confidence_score;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
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
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__confidence_score(
    const float & _arg)
  {
    this->confidence_score = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t MSG_TYPE_EMERGENCY_BRAKE =
    0u;
  static constexpr uint8_t MSG_TYPE_OBSTACLE_AHEAD =
    1u;
  static constexpr uint8_t MSG_TYPE_RECKLESS_DRIVING =
    2u;
  static constexpr uint8_t MSG_TYPE_UNKNOWN =
    255u;

  // pointer types
  using RawPtr =
    car_msgs::msg::EmergencyEvent_<ContainerAllocator> *;
  using ConstRawPtr =
    const car_msgs::msg::EmergencyEvent_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      car_msgs::msg::EmergencyEvent_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      car_msgs::msg::EmergencyEvent_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__car_msgs__msg__EmergencyEvent
    std::shared_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__car_msgs__msg__EmergencyEvent
    std::shared_ptr<car_msgs::msg::EmergencyEvent_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EmergencyEvent_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->msg_type != other.msg_type) {
      return false;
    }
    if (this->vehicle_id != other.vehicle_id) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->confidence_score != other.confidence_score) {
      return false;
    }
    return true;
  }
  bool operator!=(const EmergencyEvent_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EmergencyEvent_

// alias to use template instance with default allocator
using EmergencyEvent =
  car_msgs::msg::EmergencyEvent_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t EmergencyEvent_<ContainerAllocator>::MSG_TYPE_EMERGENCY_BRAKE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t EmergencyEvent_<ContainerAllocator>::MSG_TYPE_OBSTACLE_AHEAD;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t EmergencyEvent_<ContainerAllocator>::MSG_TYPE_RECKLESS_DRIVING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t EmergencyEvent_<ContainerAllocator>::MSG_TYPE_UNKNOWN;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace car_msgs

#endif  // CAR_MSGS__MSG__DETAIL__EMERGENCY_EVENT__STRUCT_HPP_
