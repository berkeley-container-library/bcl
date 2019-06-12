#pragma once

#include <gasnet_ratomic.h>

// NOTE: The GASNet AD Op wrapping code in this file is
//       directly borrowed from UPC++ `atomic.hpp`.

namespace BCL {

extern gex_TM_t tm;
gex_AD_t ad_i32;

template<typename T>
constexpr gex_DT_t get_gex_dt();
template<>
constexpr gex_DT_t get_gex_dt<int32_t>() { return GEX_DT_I32; }
template<>
constexpr gex_DT_t get_gex_dt<uint32_t>() { return GEX_DT_U32; }
template<>
constexpr gex_DT_t get_gex_dt<int64_t>() { return GEX_DT_I64; }
template<>
constexpr gex_DT_t get_gex_dt<uint64_t>() { return GEX_DT_U64; }

template<typename T>
gex_Event_t shim_gex_AD_OpNB(
    gex_AD_t ad, T *p, size_t rank, void *addr,
    int op, T val1, T val2, int flags
  );

template<>
gex_Event_t shim_gex_AD_OpNB<int32_t>(
    gex_AD_t ad, int32_t *p, size_t rank, void *addr,
    int op, int32_t val1, int32_t val2, int flags
  ) {
  return gex_AD_OpNB_I32(ad, p, rank, addr, op, val1, val2, flags);
}

template<>
gex_Event_t shim_gex_AD_OpNB<uint32_t>(
    gex_AD_t ad, uint32_t *p, size_t rank, void *addr,
    int op, uint32_t val1, uint32_t val2, int flags
  ) {
  return gex_AD_OpNB_U32(ad, p, rank, addr, op, val1, val2, flags);
}

template<>
gex_Event_t shim_gex_AD_OpNB<int64_t>(
    gex_AD_t ad, int64_t *p, size_t rank, void *addr,
    int op, int64_t val1, int64_t val2, int flags
  ) {
  return gex_AD_OpNB_I64(ad, p, rank, addr, op, val1, val2, flags);
}

template<>
gex_Event_t shim_gex_AD_OpNB<uint64_t>(
    gex_AD_t ad, uint64_t *p, size_t rank, void *addr,
    int op, uint64_t val1, uint64_t val2, int flags
  ) {
  return gex_AD_OpNB_U64(ad, p, rank, addr, op, val1, val2, flags);
}

void init_atomics() {
  gex_OP_t ops = GEX_OP_FADD | GEX_OP_FCAS | GEX_OP_GET | GEX_OP_FXOR | GEX_OP_FOR | GEX_OP_FAND;
  gex_Flags_t flags = 0;
  gex_AD_Create(&ad_i32, tm, get_gex_dt<int32_t>(), ops, flags);
}

void finalize_atomics() {
  gex_AD_Destroy(ad_i32);
}

}
