#pragma once

namespace BCL {

template <typename T>
T compare_and_swap(BCL::GlobalPtr<T> ptr, T old_val, T new_val) {
  static_assert(std::is_same<T, int32_t>::value);
  void* dst_ptr = gasnet_resolve_address(ptr);
  int32_t rv;
  gex_Event_t event = gex_AD_OpNB_I32(ad_i32, &rv, ptr.rank, dst_ptr,
                                      GEX_OP_FCAS, old_val, new_val, 0);

  gex_Event_Wait(event);
  return rv;
}

}
