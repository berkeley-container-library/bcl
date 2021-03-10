
#include <bcl/containers/experimental/cuda/util/error.cuh>

namespace BCL {

namespace cuda {

template <typename T>
__device__ __forceinline__ T combiner(T a, T b) {
  return a*b;
}

template <typename T>
__device__ __forceinline__ T reducer(T a, T b) {
  return a+b;
}

template <typename T>
__device__ __forceinline__ T initializer() {
  return 0;
}

template<typename T>
__global__ void spmm_test2(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    T *val_sh = (T *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<6)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        T val;
        T acc1 = initializer<T>();
        T acc2 = initializer<T>();

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    // MODIFY
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    // MODIFY (all)
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    val = val_sh[(shmem_offset+kk)];
                    // acc1 += val*B_dnVal[offset];
                    acc1 = reducer(acc1, combiner(val, B_dnVal[offset]));
                    // acc2 += val*B_dnVal[offset+32];
                    acc2 = reducer(acc2, combiner(val, B_dnVal[offset+32]));
                }
                __syncwarp();
            }
            // MODIFY (C, all)
            offset = rid*B_ncols+cid;
            C_dnVal[offset] = reducer(C_dnVal[offset], acc1);
            C_dnVal[offset+32] = reducer(C_dnVal[offset+32], acc2);
        }
        else {
            int nout = (B_ncols-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    // MODIFY
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    val = val_sh[(shmem_offset+kk)];
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    // MODIFY (all)
                    if (nout>0) {
                        // acc1 += val*B_dnVal[offset];
                        acc1 = reducer(acc1, combiner(val, B_dnVal[offset]));
                    }
                    if (nout>1) {
                        // acc2 += val*B_dnVal[offset+32];  
                        acc2 = reducer(acc2, combiner(val, B_dnVal[offset+32]));
                    }
                }
                __syncwarp();
            }
            // MODIFY (C, all)
            offset = rid*B_ncols+cid;
            if (nout>0) {
                C_dnVal[offset] = reducer(C_dnVal[offset], acc1);
            }
            if (nout>1) {
                C_dnVal[(offset+32)] = reducer(C_dnVal[offset+32], acc2);
            }
        }
    }
}

// QUESTION: what to put for tile_row?
void spmmWrapper(int tile_row, int A_nrows, int B_ncols, int *A_rowPtr, int *A_colInd, float *A_val, float *B, float *C, int b_ld, int c_ld) {
  spmm_test2<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+63)/64, 1), dim3(32, tile_row, 1), 32*tile_row*(sizeof(int)+sizeof(float)),0>>> (
      A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
  );
  CUDA_CHECK(cudaGetLastError());
}

template <typename AMatrixType, typename BMatrixType, typename CMatrixType>
void spmm_gespmm(AMatrixType& a,
                BMatrixType& b,
                CMatrixType& c)
{
  if (a.nnz() == 0) {
    return;
  }
  static_assert(std::is_same<typename BMatrixType::indexing_type, RowMajorIndexing>::value);
  static_assert(std::is_same<typename CMatrixType::indexing_type, RowMajorIndexing>::value);
  // GeSPMM does not support arbitrary LD
  assert(b.ld() == b.n());
  assert(c.ld() == c.n());

  spmmWrapper(4, a.shape()[0], b.shape()[1],
              a.rowptr_data(), a.colind_data(), a.values_data(),
              b.data(), c.data(), b.ld(), c.ld());
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_gespmm(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile_exp({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile_exp({k_offset % a.grid_shape()[1], j});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];
          auto begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = buf_b.get();
          auto local_c = c.get_local_tile({i, j});
          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (k_+1 < a.grid_shape()[1]) {
            auto begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            auto end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          spmm_gespmm(local_a, local_b, local_c);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
        }
      }
    }
  }
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

} // end cuda

} // end BCL
