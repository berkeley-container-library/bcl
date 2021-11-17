// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <mutex>

namespace BCL {

namespace global_state {

inline std::size_t shared_segment_size;
inline void *smem_base_ptr;
inline bool bcl_finalized;

std::mutex malloc_mutex;

} // end global_state
} // end BCL