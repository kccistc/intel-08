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
// Member `src`
// Member `type`
// Member `severity`
// Member `road`
// Member `suggest`
#include "rosidl_runtime_c/string_functions.h"
// Member `ts`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
car_msgs__msg__V2VAlert__init(car_msgs__msg__V2VAlert * msg)
{
  if (!msg) {
    return false;
  }
  // ver
  // src
  if (!rosidl_runtime_c__String__init(&msg->src)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // seq
  // ts
  if (!builtin_interfaces__msg__Time__init(&msg->ts)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__init(&msg->type)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // severity
  if (!rosidl_runtime_c__String__init(&msg->severity)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // distance_m
  // road
  if (!rosidl_runtime_c__String__init(&msg->road)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // lat
  // lon
  // suggest
  if (!rosidl_runtime_c__String__init(&msg->suggest)) {
    car_msgs__msg__V2VAlert__fini(msg);
    return false;
  }
  // ttl_s
  return true;
}

void
car_msgs__msg__V2VAlert__fini(car_msgs__msg__V2VAlert * msg)
{
  if (!msg) {
    return;
  }
  // ver
  // src
  rosidl_runtime_c__String__fini(&msg->src);
  // seq
  // ts
  builtin_interfaces__msg__Time__fini(&msg->ts);
  // type
  rosidl_runtime_c__String__fini(&msg->type);
  // severity
  rosidl_runtime_c__String__fini(&msg->severity);
  // distance_m
  // road
  rosidl_runtime_c__String__fini(&msg->road);
  // lat
  // lon
  // suggest
  rosidl_runtime_c__String__fini(&msg->suggest);
  // ttl_s
}

bool
car_msgs__msg__V2VAlert__are_equal(const car_msgs__msg__V2VAlert * lhs, const car_msgs__msg__V2VAlert * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // ver
  if (lhs->ver != rhs->ver) {
    return false;
  }
  // src
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->src), &(rhs->src)))
  {
    return false;
  }
  // seq
  if (lhs->seq != rhs->seq) {
    return false;
  }
  // ts
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->ts), &(rhs->ts)))
  {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->type), &(rhs->type)))
  {
    return false;
  }
  // severity
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->severity), &(rhs->severity)))
  {
    return false;
  }
  // distance_m
  if (lhs->distance_m != rhs->distance_m) {
    return false;
  }
  // road
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->road), &(rhs->road)))
  {
    return false;
  }
  // lat
  if (lhs->lat != rhs->lat) {
    return false;
  }
  // lon
  if (lhs->lon != rhs->lon) {
    return false;
  }
  // suggest
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->suggest), &(rhs->suggest)))
  {
    return false;
  }
  // ttl_s
  if (lhs->ttl_s != rhs->ttl_s) {
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
  // ver
  output->ver = input->ver;
  // src
  if (!rosidl_runtime_c__String__copy(
      &(input->src), &(output->src)))
  {
    return false;
  }
  // seq
  output->seq = input->seq;
  // ts
  if (!builtin_interfaces__msg__Time__copy(
      &(input->ts), &(output->ts)))
  {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__copy(
      &(input->type), &(output->type)))
  {
    return false;
  }
  // severity
  if (!rosidl_runtime_c__String__copy(
      &(input->severity), &(output->severity)))
  {
    return false;
  }
  // distance_m
  output->distance_m = input->distance_m;
  // road
  if (!rosidl_runtime_c__String__copy(
      &(input->road), &(output->road)))
  {
    return false;
  }
  // lat
  output->lat = input->lat;
  // lon
  output->lon = input->lon;
  // suggest
  if (!rosidl_runtime_c__String__copy(
      &(input->suggest), &(output->suggest)))
  {
    return false;
  }
  // ttl_s
  output->ttl_s = input->ttl_s;
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
