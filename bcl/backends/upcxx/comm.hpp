#pragma once

#include "backend.hpp"

namespace BCL {

extern void barrier();
extern uint64_t rank();
extern uint64_t nprocs();

template <typename T>
void read(const GlobalPtr<T>& src, T* dst, size_t size) {
  upcxx::rget(upcxx_resolve_address(src), dst, size).wait();
}

template <typename T>
void write(const T* src, const GlobalPtr<T>& dst, size_t size) {
  upcxx::rput(src, upcxx_resolve_address(dst), size).wait();
}

template <typename T>
T broadcast(T& val, uint64_t root) {
  return upcxx::broadcast(val, root).wait();
}

template <typename T, typename Op>
T allreduce(const T& val, Op fn) {
  return upcxx::allreduce(val, fn).wait();
}

}
