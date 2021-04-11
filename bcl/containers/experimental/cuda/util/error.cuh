// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ERROR_CHECK
#define ERROR_CHECK

#define CUDA_CHECK(call) {  \
  cudaError_t err = call;   \
  if( cudaSuccess != err) { \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
         __FILE__, __LINE__, cudaGetErrorString( err) );          \
    exit(EXIT_FAILURE);     \
  }                         \ 
}                           



#endif
