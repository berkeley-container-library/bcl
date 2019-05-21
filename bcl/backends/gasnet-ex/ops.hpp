#pragma once

#include <gasnetex.h>
#include <gasnet_ratomic.h>
#include <cassert>

namespace BCL {

// TODO: use actual GASNet reduce?

template <typename T, typename Op>
void gasnet_fn_bcl_impl_(void* results, size_t result_count,
                const void* left_operands, size_t left_count,
                const void* right_operands,
                size_t elem_size, int flags, int arg) {
  T* results_ = reinterpret_cast<T*>(results);
  const T* left_operands_ = reinterpret_cast<const T*>(left_operands);
  const T* right_operands_ = reinterpret_cast<const T*>(right_operands);

  assert(result_count == 1);

  T reduce_val = right_operands_[0];
  for (size_t i = 1; i < result_count; i++) {
    reduce_val = Op{}(reduce_val, right_operands_[i]);
  }

  for (size_t i = 0; i < left_count; i++) {
    reduce_val = Op{}(reduce_val, left_operands_[i]);
  }

  *results_ = reduce_val;
}

// Ops define an MPI OP and an MPI Datatype
template <typename T>
struct abstract_op {
  virtual gex_OP_t op() const = 0;
  virtual gex_DT_t type() const = 0;
};

// Some special atomic ops also work for atomics
template <typename T>
struct atomic_op : public virtual abstract_op <T> {};

// Define datatypes
struct abstract_int : public virtual abstract_op <int32_t> {
  gex_OP_t type() const { return GEX_DT_I32; }
};

struct abstract_uint64_t : public virtual abstract_op <uint64_t> {
  gex_DT_t type() const { return GEX_DT_U64; }
};

struct abstract_float : public virtual abstract_op <float> {
  gex_DT_t type() const { return GEX_DT_FLT; }
};

struct abstract_double : public virtual abstract_op <double> {
  gex_DT_t type() const { return GEX_DT_DBL; }
};

template <typename T>
struct abstract_xor_: public virtual abstract_op<T>{};

template <typename T> struct xor_;

template <>
struct xor_<uint64_t> : public abstract_xor_<uint64_t>, public abstract_uint64_t, public atomic_op<uint64_t> {
  gex_OP_t op() const { return GEX_OP_FXOR; }
};

template <>
struct xor_<int> : public abstract_xor_<int>, public abstract_int, public atomic_op<int> {
  gex_OP_t op() const { return GEX_OP_FXOR; }
};

template <typename T>
struct abstract_or_: public virtual abstract_op<T>{};

template <typename T> struct or_;

template <>
struct or_<int> : public abstract_or_<int>, public abstract_int, public atomic_op<int> {
  gex_OP_t op() const { return GEX_OP_FOR; }
};

template <typename T>
struct abstract_and_: public virtual abstract_op<T>{};

template <typename T> struct and_;

template <>
struct and_<int> : public abstract_and_<int>, public abstract_int, public atomic_op<int> {
  gex_OP_t op() const { return GEX_OP_FAND; }
};

// Define the plus operation
template <typename T>
struct abstract_plus : public virtual abstract_op <T> {
  gex_OP_t op() const { return GEX_OP_FADD; }
};

template <typename T> struct plus;

template <>
struct plus <int> : public abstract_plus <int>, public abstract_int, public atomic_op <int> {};

template <>
struct plus <uint64_t> : public abstract_plus <uint64_t>, public abstract_uint64_t, public atomic_op <uint64_t> {};

template <>
struct plus <float> : public abstract_plus <float>, public abstract_float {};

template <>
struct plus <double> : public abstract_plus <double>, public abstract_double {};

}
