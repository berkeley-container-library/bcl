#pragma once

#include <mpp/shmem.h>

namespace BCL {
  // Ops define an MPI OP and an MPI Datatype
  template <typename T>
  struct abstract_op {};

  // Some special atomic ops also offer
  // a shmem atomic version
  template <typename T>
  struct atomic_op : public virtual abstract_op <T> {
    virtual T shmem_atomic_op(const GlobalPtr <T> ptr, const T &val) const = 0;
  };

  // Define datatypes
  struct abstract_int : public virtual abstract_op <int> {};

  struct abstract_uint64_t : public virtual abstract_op <uint64_t> {};

  struct abstract_float : public virtual abstract_op <float> {};

  struct abstract_double : public virtual abstract_op <double> {};

  template <typename T>
  struct abstract_swap : public virtual abstract_op<T>{};

  template <typename T> struct swap;

  template <>
  struct swap<uint64_t> : public abstract_swap<uint64_t>, public abstract_uint64_t, public atomic_op<uint64_t> {
    uint64_t shmem_atomic_op(const GlobalPtr<uint64_t> ptr, const uint64_t& val) const {
      return shmem_uint64_atomic_swap(ptr.rptr(), val, ptr.rank);
    }
  };

  template <typename T>
  struct abstract_xor_: public virtual abstract_op<T>{};

  template <typename T> struct xor_;

  template <>
  struct xor_<uint64_t> : public abstract_xor_<uint64_t>, public abstract_uint64_t, public atomic_op<uint64_t> {
    uint64_t shmem_atomic_op(const GlobalPtr<uint64_t> ptr, const uint64_t& val) const {
      return shmem_uint64_atomic_fetch_xor(ptr.rptr(), val, ptr.rank);
    }
  };

  template <>
  struct xor_<int> : public abstract_xor_<int>, public abstract_int, public atomic_op<int> {
    int shmem_atomic_op(const GlobalPtr<int> ptr, const int& val) const {
      return shmem_int32_atomic_fetch_xor(ptr.rptr(), val, ptr.rank);
    }
  };

  template <typename T>
  struct abstract_or_: public virtual abstract_op<T>{};

  template <typename T> struct or_;

  template <>
  struct or_<int> : public abstract_or_<int>, public abstract_int, public atomic_op<int> {
    int shmem_atomic_op(const GlobalPtr<int> ptr, const int& val) const {
      return shmem_int32_atomic_fetch_or(ptr.rptr(), val, ptr.rank);
    }
  };

  template <typename T>
  struct abstract_and_: public virtual abstract_op<T>{};

  template <typename T> struct and_;

  template <>
  struct and_<int> : public abstract_and_<int>, public abstract_int, public atomic_op<int> {
    int shmem_atomic_op(const GlobalPtr<int> ptr, const int& val) const {
      return shmem_int32_atomic_fetch_and(ptr.rptr(), val, ptr.rank);
    }
  };

  // Define the plus operation
  template <typename T>
  struct abstract_plus : public virtual abstract_op <T> {};

  template <typename T> struct plus;

  template <>
  struct plus <uint64_t> : public abstract_plus <uint64_t>, public abstract_uint64_t {};

  template <>
  struct plus <int> : public abstract_plus <int>, public abstract_int, public atomic_op <int> {
    int shmem_atomic_op(const GlobalPtr <int> ptr, const int &val) const {
      return shmem_int_fadd(ptr.rptr(), val, ptr.rank);
    }
  };

  template <>
  struct plus <float> : public abstract_plus <float>, public abstract_float {};

  template <>
  struct plus <double> : public abstract_plus <double>, public abstract_double {};

  template <typename T>
  struct abstract_land : public virtual abstract_op <T> {
  };

  template <typename T>
  struct abstract_max : public virtual abstract_op <T> {
  };

  template <typename T> struct land;
  template <typename T> struct max;

  template <>
  struct land <int> : public abstract_land <int>, public abstract_int {};

  template <>
  struct max <int> : public abstract_max <int>, public abstract_int {};
}
