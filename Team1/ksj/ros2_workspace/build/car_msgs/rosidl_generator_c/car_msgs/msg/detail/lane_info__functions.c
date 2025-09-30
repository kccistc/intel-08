// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from car_msgs:msg/LaneInfo.idl
// generated code does not contain a copyright notice
#include "car_msgs/msg/detail/lane_info__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
car_msgs__msg__LaneInfo__init(car_msgs__msg__LaneInfo * msg)
{
  if (!msg) {
    return false;
  }
  // is_detected
  // curvature
  // offset
  return true;
}

void
car_msgs__msg__LaneInfo__fini(car_msgs__msg__LaneInfo * msg)
{
  if (!msg) {
    return;
  }
  // is_detected
  // curvature
  // offset
}

bool
car_msgs__msg__LaneInfo__are_equal(const car_msgs__msg__LaneInfo * lhs, const car_msgs__msg__LaneInfo * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // is_detected
  if (lhs->is_detected != rhs->is_detected) {
    return false;
  }
  // curvature
  if (lhs->curvature != rhs->curvature) {
    return false;
  }
  // offset
  if (lhs->offset != rhs->offset) {
    return false;
  }
  return true;
}

bool
car_msgs__msg__LaneInfo__copy(
  const car_msgs__msg__LaneInfo * input,
  car_msgs__msg__LaneInfo * output)
{
  if (!input || !output) {
    return false;
  }
  // is_detected
  output->is_detected = input->is_detected;
  // curvature
  output->curvature = input->curvature;
  // offset
  output->offset = input->offset;
  return true;
}

car_msgs__msg__LaneInfo *
car_msgs__msg__LaneInfo__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__LaneInfo * msg = (car_msgs__msg__LaneInfo *)allocator.allocate(sizeof(car_msgs__msg__LaneInfo), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(car_msgs__msg__LaneInfo));
  bool success = car_msgs__msg__LaneInfo__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
car_msgs__msg__LaneInfo__destroy(car_msgs__msg__LaneInfo * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    car_msgs__msg__LaneInfo__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
car_msgs__msg__LaneInfo__Sequence__init(car_msgs__msg__LaneInfo__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__LaneInfo * data = NULL;

  if (size) {
    data = (car_msgs__msg__LaneInfo *)allocator.zero_allocate(size, sizeof(car_msgs__msg__LaneInfo), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = car_msgs__msg__LaneInfo__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        car_msgs__msg__LaneInfo__fini(&data[i - 1]);
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
car_msgs__msg__LaneInfo__Sequence__fini(car_msgs__msg__LaneInfo__Sequence * array)
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
      car_msgs__msg__LaneInfo__fini(&array->data[i]);
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

car_msgs__msg__LaneInfo__Sequence *
car_msgs__msg__LaneInfo__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  car_msgs__msg__LaneInfo__Sequence * array = (car_msgs__msg__LaneInfo__Sequence *)allocator.allocate(sizeof(car_msgs__msg__LaneInfo__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = car_msgs__msg__LaneInfo__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
car_msgs__msg__LaneInfo__Sequence__destroy(car_msgs__msg__LaneInfo__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    car_msgs__msg__LaneInfo__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
car_msgs__msg__LaneInfo__Sequence__are_equal(const car_msgs__msg__LaneInfo__Sequence * lhs, const car_msgs__msg__LaneInfo__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!car_msgs__msg__LaneInfo__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
car_msgs__msg__LaneInfo__Sequence__copy(
  const car_msgs__msg__LaneInfo__Sequence * input,
  car_msgs__msg__LaneInfo__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(car_msgs__msg__LaneInfo);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    car_msgs__msg__LaneInfo * data =
      (car_msgs__msg__LaneInfo *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!car_msgs__msg__LaneInfo__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          car_msgs__msg__LaneInfo__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!car_msgs__msg__LaneInfo__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
