// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

template<typename T>  
struct MyHash{
  __device__ __host__ uint32_t operator()(T key, uint32_t seed)
  {
    return key;
  }
};
