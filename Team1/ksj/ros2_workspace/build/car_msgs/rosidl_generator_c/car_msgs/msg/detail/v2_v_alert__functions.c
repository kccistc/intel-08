// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from car_msgs:msg/V2VAlert.idl
// generated code does not contain a copyright notice
#include "car_msgs/msg/detail/v2_v_alert__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `vehicle_id`
#include "rosidl_runtime_c/string_functions.h"

bool
car_msgs__msg__V2VAlert__init(car_msgs__msg__V2VAlert * msg)
{
  if (!msg) {
    return false;
  }
  // msg_type
  // vehicle_id
  if (!rosidl_runtime_c__String__init(&msg->vehicle_id)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // distance
  return true;
}

void
car_msgs__msg__V2VAlert__fini(car_msgs__msg__V2VAlert * msg)
{
  if (!msg) {
    return;
  }
  // msg_type
  // vehicle_id
  rosidl_runtime_c__String__fini(&msg->vehicle_id);
  // distance
}

bool
car_msgs__msg__V2VAlert__are_equal(const car_msgs__msg__V2VAlert * lhs, const car_msgs__msg__V2VAlert * rhs)
{
  if (!lhs || !rhs) {
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
  // distance
  if (lhs->distance != rhs->distance) {
    return false;
  }
  return true;
}

bool
car_msgs__msg__V2VAlert__copy(
  const car_msgs__msg__V2VAlert * input,
  car_msgs__msg__V2VAlert * output)
{
  if (!input || !output) {
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
  // distance
  output->distance = input->distance;
  return true;
}

car_msgs__msg__V2VAlert *
car_msgs__msg__V2VAlert__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__V2VAlert * msg = (car_msgs__msg__V2VAlert *)allocator.allocate(sizeof(car_msgs__msg__V2VAlert), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(car_msgs__msg__V2VAlert));
  bool success = car_msgs__msg__V2VAlert__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
car_msgs__msg__V2VAlert__destroy(car_msgs__msg__V2VAlert * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    car_msgs__msg__V2VAlert__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
car_msgs__msg__V2VAlert__Sequence__init(car_msgs__msg__V2VAlert__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__V2VAlert * data = NULL;

  if (size) {
    data = (car_msgs__msg__V2VAlert *)allocator.zero_allocate(size, sizeof(car_msgs__msg__V2VAlert), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = car_msgs__msg__V2VAlert__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        car_msgs__msg__V2VAlert__fini(&data[i - 1]);
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
car_msgs__msg__V2VAlert__Sequence__fini(car_msgs__msg__V2VAlert__Sequence * array)
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
      car_msgs__msg__V2VAlert__fini(&array->data[i]);
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

car_msgs__msg__V2VAlert__Sequence *
car_msgs__msg__V2VAlert__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__V2VAlert__Sequence * array = (car_msgs__msg__V2VAlert__Sequence *)allocator.allocate(sizeof(car_msgs__msg__V2VAlert__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = car_msgs__msg__V2VAlert__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
car_msgs__msg__V2VAlert__Sequence__destroy(car_msgs__msg__V2VAlert__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    car_msgs__msg__V2VAlert__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
car_msgs__msg__V2VAlert__Sequence__are_equal(const car_msgs__msg__V2VAlert__Sequence * lhs, const car_msgs__msg__V2VAlert__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!car_msgs__msg__V2VAlert__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
car_msgs__msg__V2VAlert__Sequence__copy(
  const car_msgs__msg__V2VAlert__Sequence * input,
  car_msgs__msg__V2VAlert__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(car_msgs__msg__V2VAlert);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    car_msgs__msg__V2VAlert * data =
      (car_msgs__msg__V2VAlert *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!car_msgs__msg__V2VAlert__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          car_msgs__msg__V2VAlert__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!car_msgs__msg__V2VAlert__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
