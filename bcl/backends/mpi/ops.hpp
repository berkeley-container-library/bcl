#pragma once

#include <mpi.h>

namespace BCL {
  // Ops define an MPI OP and an MPI Datatype
  template <typename T>
  struct abstract_op {
    virtual MPI_Op op() const = 0;
    virtual MPI_Datatype type() const = 0;
  };

  // Some special atomic ops also work for atomics
  template <typename T>
  struct atomic_op : public virtual abstract_op <T> {};

  // Define datatypes
  struct abstract_int : public virtual abstract_op <int> {
    MPI_Datatype type() const { return MPI_INT; }
  };

  struct abstract_uint64_t : public virtual abstract_op <uint64_t> {
    MPI_Datatype type() const { return MPI_UNSIGNED_LONG_LONG; }
  };

  struct abstract_float : public virtual abstract_op <float> {
    MPI_Datatype type() const { return MPI_FLOAT; }
  };

  struct abstract_double : public virtual abstract_op <double> {
    MPI_Datatype type() const { return MPI_DOUBLE; }
  };

  template <typename T>
  struct abstract_or_: public virtual abstract_op<T>{};

  template <typename T> struct or_;

  template <>
  struct or_<int> : public abstract_or_<int>, public abstract_int, public atomic_op<int> {
    MPI_Op op() const { return MPI_BOR; }
  };

  template <typename T>
  struct abstract_xor_: public virtual abstract_op<T>{};

  template <typename T> struct xor_;

  template <>
  struct xor_<uint64_t> : public abstract_xor_<uint64_t>, public abstract_uint64_t, public atomic_op<uint64_t> {
    MPI_Op op() const { return MPI_BXOR; }
  };

  template <>
  struct xor_<int> : public abstract_xor_<int>, public abstract_int, public atomic_op<int> {
    MPI_Op op() const { return MPI_BXOR; }
  };

  template <typename T>
  struct abstract_and_: public virtual abstract_op<T>{};

  template <typename T> struct and_;

  template <>
  struct and_<int> : public abstract_and_<int>, public abstract_int, public atomic_op<int> {
    MPI_Op op() const { return MPI_BAND; }
  };

  // Define the plus operation
  template <typename T>
  struct abstract_plus : public virtual abstract_op <T> {
    MPI_Op op() const { return MPI_SUM; }
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

  template <typename T>
  struct abstract_land : public virtual abstract_op <T> {
    MPI_Op op() const { return MPI_LAND; }
  };

  template <typename T> struct land;

  template <>
  struct land <int> : public abstract_land <int>, public abstract_int {};
}
