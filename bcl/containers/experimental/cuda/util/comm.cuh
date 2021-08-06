// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef COMMON
#define COMMON

#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define LANE (threadIdx.x&31)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)

__device__ uint32_t flagMask0 = 3;
__device__ uint32_t flagMask1 = 768;
__device__ uint32_t flagMask2 = 196608;
__device__ uint32_t flagMask3 = 50331648;
__device__ uint32_t flagMask4 = 1;
__device__ uint32_t flagMask5 = 65536;

#define align_up(num, align) \
  (((num) + ((align) - 1)) & ~((align) - 1))

#define align_down(num, align) \
  ((num) & ~((align) - 1))

#endif
