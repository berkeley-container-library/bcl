#define __thrust_compiler_fence() __sync_synchronize()
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <thrust/sort.h>

#include <bcl/containers/experimental/cuda/CudaSPMatrix.hpp>

#include <unordered_map>

#include "cusparse_util.hpp"
#include "spgemm.hpp"
#include "cusp_util.hpp"

#include <chrono>
#include <essl.h>

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
  return T();
}

template<typename T>
__global__ void spmm_test2(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal,
    int b_ld, int c_ld
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
            C_dnVal[offset] = acc1;
            C_dnVal[offset+32] = acc2;
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
                    acc2 = reducer(acc1, combiner(val, B_dnVal[offset+32]));
                    }
                }
                __syncwarp();
            }
            // MODIFY (C, all)
            offset = rid*B_ncols+cid;
            if (nout>0) {
            C_dnVal[offset] = acc1;
            }
            if (nout>1) {
            C_dnVal[(offset+32)] = acc2;
            }
        }
    }
}

// QUESTION: what to put for tile_row?
void spmmWrapper(int tile_row, int A_nrows, int B_ncols, int *A_rowPtr, int *A_colInd, float *A_val, float *B, float *C, int b_ld, int c_ld) {
  spmm_test2<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+63)/64, 1), dim3(32, tile_row, 1), 32*tile_row*(sizeof(int)+sizeof(float)),0>>> (
      A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C, b_ld, c_ld
  );
  cudaError_t error;
  error = cudaGetLastError();
  CUDA_CHECK(error);
}

namespace BCL {
namespace cuda {

template <typename AMatrixType, typename BMatrixType, typename CMatrixType>
void spmm_gspmm(AMatrixType& a,
                BMatrixType& b,
                CMatrixType& c)
{
  if (a.nnz() == 0) {
    return;
  }

  spmmWrapper(4, a.shape()[0], b.shape()[1],
              a.rowptr_data(), a.colind_data(), a.values_data(),
              b.data(), c.data(), b.ld(), c.ld());
}
}
}

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init();

  using T = float;

  using allocator_type = BCL::cuda::bcl_allocator<T>;

  std::string fname = "/autofs/nccs-svm1_home2/b2v/pkg/cusplibrary/examples/Matmul/chesapeake_general.mtx";

  BCL::CSRMatrix<T, graphblas::Index> mat(fname);

  auto local_a = BCL::cuda::to_gpu<T, graphblas::Index, allocator_type>(mat);

  size_t m = local_a.shape()[0];
  size_t k = local_a.shape()[1];

  size_t n = 8;

  BCL::cuda::CudaMatrix<T, allocator_type> local_b({k, n});
  BCL::cuda::CudaMatrix<T, allocator_type> local_c_cusparse({m, n});
  BCL::cuda::CudaMatrix<T, allocator_type> local_c_gspmm({m, n});

  local_b = 1;
  local_c_cusparse = 0;
  local_c_gspmm = 0;

  BCL::cuda::spmm_cusparse(local_a, local_b, local_c_cusparse);

  BCL::cuda::spmm_gspmm(local_a, local_b, local_c_gspmm);

  std::vector<T> cusparse_data(local_c_cusparse.size());
  cudaMemcpy(cusparse_data.data(), local_c_cusparse.data(), cusparse_data.size()*sizeof(T), cudaMemcpyDeviceToHost);

  std::vector<T> gspmm_data(local_c_gspmm.size());
  cudaMemcpy(gspmm_data.data(), local_c_gspmm.data(), gspmm_data.size()*sizeof(T), cudaMemcpyDeviceToHost);

  bool print = false;
  T eps = 1.0e-5;
  size_t matching = 0;

  for (size_t i = 0; i < local_c_gspmm.shape()[0]; i++) {
    for (size_t j = 0; j < local_c_gspmm.shape()[1]; j++) {
      size_t idx = i + j*local_c_gspmm.shape()[0];
      size_t cusp_idx = i*local_c_gspmm.shape()[1] + j;
      if (std::abs(cusparse_data[idx] - gspmm_data[cusp_idx]) > eps) {
        assert(false);
        if (print) {
          printf("O %2.2lf != %2.2lf ", cusparse_data[idx], gspmm_data[cusp_idx]);
        }
      } else {
        if (print) {
          printf("X %2.2lf == %2.2lf ", cusparse_data[idx], gspmm_data[cusp_idx]);
        }
        matching++;
      }
    }
    if (print) {
      printf("\n");
    }
  }

  BCL::finalize();

  return 0;
}
