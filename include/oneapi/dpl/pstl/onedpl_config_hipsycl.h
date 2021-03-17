// -*- C++ -*-
//===-- utils.h -----------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

/// \brief This file establishes compatibility for the DPC++-specific parts of
/// the oneDPL code base with hipSYCL/SYCL 2020.

#ifndef ONEDPL_CONFIG_HIPSYCL_HPP
#define ONEDPL_CONFIG_HIPSYCL_HPP

#    include "CL/sycl.hpp"
using namespace cl;

#     include <cstdint>
// oneDPL needs legacy OpenCL types
namespace hipsycl::sycl {
using cl_char = char;
using cl_uchar = unsigned char;
using cl_short = int16_t;
using cl_ushort = uint16_t;
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_long = int64_t;
using cl_ulong = uint64_t;
using cl_float = float;
using cl_double = double;

// Not part of SYCL 2020, but implementation detail
// of DPC++ that oneDPL needs.
template <access_mode mode> struct mode_tag_t {
  explicit mode_tag_t() = default;
};

namespace ONEAPI {
using namespace ::hipsycl::sycl;
}

namespace property {
// oneDPL requires older noinit name, while hipSYCL provides no_init
// as per SYCL 2020 final spec.
using noinit = no_init;
}

// DPC++ pre-SYCL 2020 final group algorithm compatibility


template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
bool any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_any_of(g, first, last, pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
bool all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_all_of(g, first, last, pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
bool none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_none_of(g, first, last, pred);
}

template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
typename std::iterator_traits<Ptr>::value_type
reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  return joint_reduce(g, first, last, binary_op);
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
T reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  return joint_reduce(g, first, last, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
OutPtr exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                      BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
T exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                 BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
OutPtr inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                      BinaryOperation binary_op) {
  return joint_inclusive_scan(g, first, last, result, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
T inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                 BinaryOperation binary_op, T init) {
  return joint_inclusive_scan(g, first, last, result, binary_op, init);
}

template <typename Group, typename T, typename Predicate>
bool any_of(Group g, T x, Predicate pred) {
  return group_any_of(g, x, pred);
}

template <typename Group>
bool any_of(Group g, bool pred) {
  return group_any_of(g, pred);
}

template <typename Group, typename T, typename Predicate>
bool all_of(Group g, T x, Predicate pred) {
  return group_all_of(g, x, pred);
}

template <typename Group>
bool all_of(Group g, bool pred) {
  return group_all_of(g, pred);
}

template <typename Group, typename T, typename Predicate>
bool none_of(Group g, T x, Predicate pred) {
  return group_none_of(g, x, pred);
}

template <typename Group>
bool none_of(Group g, bool pred) {
  return group_none_of(g, pred);
}

template <typename Group, typename T, typename BinaryOperation>
T reduce(Group g, T x, BinaryOperation binary_op) {
  return group_reduce(g, x, binary_op);
}

template <typename Group, typename V, typename T, typename BinaryOperation>
T reduce(Group g, V x, T init, BinaryOperation binary_op){
  return group_reduce(g, x, init, binary_op);
}

template <typename Group, typename T, typename BinaryOperation>
T exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return group_exclusive_scan(g, x, binary_op);
}

template <typename Group, typename V, typename T, typename BinaryOperation>
T exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return group_exclusive_scan(g, x, init, binary_op);
}

template <typename Group, typename T, typename BinaryOperation>
T inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return group_inclusive_scan(g, x, binary_op);
}

template <typename Group, typename V, typename T, typename BinaryOperation>
T inclusive_scan(Group g, V x, BinaryOperation binary_op, T init) {
  return group_inclusive_scan(g, x, init, binary_op);
}

}
#define sycl_buffer_allocator(T) sycl::buffer_allocator<T>

#endif
