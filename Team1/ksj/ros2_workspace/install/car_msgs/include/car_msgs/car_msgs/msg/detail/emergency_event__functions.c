// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from car_msgs:msg/EmergencyEvent.idl
// generated code does not contain a copyright notice
#include "car_msgs/msg/detail/emergency_event__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `vehicle_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
car_msgs__msg__EmergencyEvent__init(car_msgs__msg__EmergencyEvent * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    car_msgs__msg__EmergencyEvent__fini(msg);
    return false;
  }
  // msg_type
  // vehicle_id
  if (!rosidl_runtime_c__String__init(&msg->vehicle_id)) {
    car_msgs__msg__EmergencyEvent__fini(msg);
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    car_msgs__msg__EmergencyEvent__fini(msg);
    return false;
  }
  // confidence_score
  return true;
}

void
car_msgs__msg__EmergencyEvent__fini(car_msgs__msg__EmergencyEvent * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // msg_type
  // vehicle_id
  rosidl_runtime_c__String__fini(&msg->vehicle_id);
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // confidence_score
}

bool
car_msgs__msg__EmergencyEvent__are_equal(const car_msgs__msg__EmergencyEvent * lhs, const car_msgs__msg__EmergencyEvent * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // msg_type
  if (lhs->msg_type != rhs->msg_type) {
    return false;
  }
  // vehicle_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->vehicle_id), &(rhs->vehicle_id)))
  {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // confidence_score
  if (lhs->confidence_score != rhs->confidence_score) {
    return false;
  }
  return true;
}

bool
car_msgs__msg__EmergencyEvent__copy(
  const car_msgs__msg__EmergencyEvent * input,
  car_msgs__msg__EmergencyEvent * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // msg_type
  output->msg_type = input->msg_type;
  // vehicle_id
  if (!rosidl_runtime_c__String__copy(
      &(input->vehicle_id), &(output->vehicle_id)))
  {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // confidence_score
  output->confidence_score = input->confidence_score;
  return true;
}

car_msgs__msg__EmergencyEvent *
car_msgs__msg__EmergencyEvent__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__EmergencyEvent * msg = (car_msgs__msg__EmergencyEvent *)allocator.allocate(sizeof(car_msgs__msg__EmergencyEvent), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(car_msgs__msg__EmergencyEvent));
  bool success = car_msgs__msg__EmergencyEvent__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
car_msgs__msg__EmergencyEvent__destroy(car_msgs__msg__EmergencyEvent * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    car_msgs__msg__EmergencyEvent__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
car_msgs__msg__EmergencyEvent__Sequence__init(car_msgs__msg__EmergencyEvent__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__EmergencyEvent * data = NULL;

  if (size) {
    data = (car_msgs__msg__EmergencyEvent *)allocator.zero_allocate(size, sizeof(car_msgs__msg__EmergencyEvent), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = car_msgs__msg__EmergencyEvent__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        car_msgs__msg__EmergencyEvent__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
car_msgs__msg__EmergencyEvent__Sequence__fini(car_msgs__msg__EmergencyEvent__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      car_msgs__msg__EmergencyEvent__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

car_msgs__msg__EmergencyEvent__Sequence *
car_msgs__msg__EmergencyEvent__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__EmergencyEvent__Sequence * array = (car_msgs__msg__EmergencyEvent__Sequence *)allocator.allocate(sizeof(car_msgs__msg__EmergencyEvent__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = car_msgs__msg__EmergencyEvent__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
car_msgs__msg__EmergencyEvent__Sequence__destroy(car_msgs__msg__EmergencyEvent__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    car_msgs__msg__EmergencyEvent__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
car_msgs__msg__EmergencyEvent__Sequence__are_equal(const car_msgs__msg__EmergencyEvent__Sequence * lhs, const car_msgs__msg__EmergencyEvent__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!car_msgs__msg__EmergencyEvent__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
car_msgs__msg__EmergencyEvent__Sequence__copy(
  const car_msgs__msg__EmergencyEvent__Sequence * input,
  car_msgs__msg__EmergencyEvent__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(car_msgs__msg__EmergencyEvent);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    car_msgs__msg__EmergencyEvent * data =
      (car_msgs__msg__EmergencyEvent *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!car_msgs__msg__EmergencyEvent__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          car_msgs__msg__EmergencyEvent__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!car_msgs__msg__EmergencyEvent__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
