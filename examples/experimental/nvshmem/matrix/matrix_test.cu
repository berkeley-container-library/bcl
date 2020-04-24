
#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include "cublas_v2.h"

#include <chrono>
#include <essl.h>

double num_gflops(size_t M, size_t N, size_t K) {
    return 1e-9 * (2*M*N*K + 3*M*N);
}

cublasHandle_t cublas_handle;

template <typename T>
void print_vector(const std::vector<T>& vec, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      std::cout << vec[i*n + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void dgemm_simple(BCL::cuda::Matrix<T>& a, BCL::cuda::Matrix<T>& b, BCL::cuda::Matrix<T>& c) {
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (BCL::rank() == c.tile_ptr({i, j}).rank_) {
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          auto a_local = a.get_tile({i, k});
          auto b_local = b.get_tile({k, j});

          T* c_ptr = c.tile_ptr({i, j}).local();

          cublasOperation_t transa = CUBLAS_OP_N;
          cublasOperation_t transb = CUBLAS_OP_N;

          T alpha = 1.0;
          T beta = 1.0;
          cublasDgemm(cublas_handle, transa, transb,
                      c.tile_shape({i, j})[0], // m
                      c.tile_shape({i, j})[1], // n
                      a.tile_shape({i, k})[1], // k
                      &alpha,
                      a_local.data(), a.tile_shape()[0],
                      b_local.data(), b.tile_shape()[0],
                      &beta,
                      c_ptr, c.tile_shape()[0]);
          cudaDeviceSynchronize();

          a_local.destroy();
          b_local.destroy();
        }
      }
    }
  }
  BCL::cuda::barrier();
}

cublasStatus_t cublasGemmWrapper(cublasHandle_t handle,
                                 cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n, int k,
                                 const float           *alpha,
                                 const float           *A, int lda,
                                 const float           *B, int ldb,
                                 const float           *beta,
                                 float                 *C, int ldc)
{
  return cublasSgemm(handle, transa, transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}

cublasStatus_t cublasGemmWrapper(cublasHandle_t handle,
                                 cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n, int k,
                                 const double           *alpha,
                                 const double           *A, int lda,
                                 const double           *B, int ldb,
                                 const double           *beta,
                                 double                 *C, int ldc)
{
  return cublasDgemm(handle, transa, transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}

void cblas_gemm(const CBLAS_LAYOUT Layout,
                const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                const int m, const int n, const int k,
                const double alpha,
                const double *a, const int lda,
                const double *b, const int ldb,
                const double beta,
                double *c, const int ldc)
{
  cblas_dgemm(Layout, transa, transb,
              m, n, k,
              alpha,
              a, lda,
              b, ldb,
              beta,
              c, ldc);
}

void cblas_gemm(const CBLAS_LAYOUT Layout,
                const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                const int m, const int n, const int k,
                const float alpha,
                const float *a, const int lda,
                const float *b, const int ldb,
                const float beta,
                float *c, const int ldc)
{
  cblas_sgemm(Layout, transa, transb,
              m, n, k,
              alpha,
              a, lda,
              b, ldb,
              beta,
              c, ldc);
}



double duration_issue;
double duration_sync;
double duration_compute;
double duration_barrier;

template <typename T>
void gemm(BCL::cuda::Matrix<T>& a, BCL::cuda::Matrix<T>& b, BCL::cuda::Matrix<T>& c) {
  using timer_type = decltype(std::chrono::high_resolution_clock::now());
  timer_type fetch_begin, fetch_end;
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (BCL::rank() == c.tile_ptr({i, j}).rank_) {
        size_t k_offset = i + j;
        auto begin = std::chrono::high_resolution_clock::now();
        fetch_begin = begin;
        auto buf_a = a.arget_tile({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile({k_offset % a.grid_shape()[1], j});
        /*
        printf("RANK(%lu) first get is from (%lu, %lu)\n", BCL::rank(),
                            a.tile_ptr({i, k_offset % a.grid_shape()[1]}).rank_,
                            b.tile_ptr({k_offset % a.grid_shape()[1], j}));
                            */
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto begin = std::chrono::high_resolution_clock::now();
          auto a_local = buf_a.get();
          auto b_local = buf_b.get();
          auto end = std::chrono::high_resolution_clock::now();
          fetch_end = end;
          duration_sync += std::chrono::duration<double>(end - begin).count();

          
          double words_fetched = sizeof(T)*(a.tile_size() + b.tile_size());
          words_fetched *= 1e-9;
          double gbps = words_fetched / std::chrono::duration<double>(fetch_end - fetch_begin).count();
          /*
          printf("RANK(%lu) %lu'th sync is %lf (%lf GB/s, %s)\n", BCL::rank(), k_,
                 std::chrono::duration<double>(end - begin).count(),
                 gbps, (gbps < 4.17) ? "SLOW" : "FAST");
                 */

          T* c_ptr = c.tile_ptr({i, j}).local();

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            fetch_begin = begin;
            buf_a = a.arget_tile({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile({(k+1) % a.grid_shape()[1], j});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          cublasOperation_t transa = CUBLAS_OP_N;
          cublasOperation_t transb = CUBLAS_OP_N;

          T alpha = 1.0;
          T beta = 1.0;
          begin = std::chrono::high_resolution_clock::now();
          cublasGemmWrapper(cublas_handle, transa, transb,
                            c.tile_shape({i, j})[0], // m
                            c.tile_shape({i, j})[1], // n
                            a.tile_shape({i, k})[1], // k
                            &alpha,
                            a_local.data(), a.tile_shape()[0],
                            b_local.data(), b.tile_shape()[0],
                            &beta,
                            c_ptr, c.tile_shape()[0]);
          cudaDeviceSynchronize();
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
          double gflops = num_gflops(c.tile_shape({i, j})[0], c.tile_shape({i, j})[1], a.tile_shape({i, k})[1])
                          / std::chrono::duration<double>(end - begin).count();
                          /*
          printf("RANK(%lu) Local matmul %lf GFLOPs (%lf of peak)\n",
                 BCL::rank(), gflops, (100*(1e-3*gflops / 15.7)));
                 */

          a_local.destroy();
          b_local.destroy();
        }
      }
    }
  }
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  duration_barrier = std::chrono::duration<double>(end - begin).count();
}

int main(int argc, char** argv) {
  BCL::init(16);

  BCL::cuda::init(8000);

  cublasCreate(&cublas_handle);

  BCL::print("Getting dims...\n");

  size_t dim = std::atol(argv[1]);

  using matmul_type = float;

  duration_issue = 0;
  duration_sync = 0;
  duration_compute = 0;
  duration_barrier = 0;

  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(dim, dim, dim);

  BCL::print("Creating matrices...\n");
  BCL::cuda::Matrix<matmul_type> a(dim, dim, std::move(blocks[0]));
  BCL::cuda::Matrix<matmul_type> b(dim, dim, std::move(blocks[1]));
  BCL::cuda::Matrix<matmul_type> c(dim, dim, std::move(blocks[2]));

  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    a.print_info();
    b.print_info();
    c.print_info();
  }

  bool print_matrices = false;
  bool test_result = false;

  // a.apply_by_index([] __device__ (auto value, auto i, auto j) { return 2; });
  // b.apply_by_index([] __device__ (auto value, auto i, auto j) { return 3; });
  BCL::print("Initialize matrices...\n");
  srand48(BCL::rank());
  a.randomize();
  b.randomize();
  c.apply([] __device__ (auto m) { return 0; });

  BCL::cuda::barrier();

  BCL::print("Print matrices...\n");
  if (BCL::rank() == 0 && print_matrices) {
    auto local_a = a.get_matrix();
    auto local_b = b.get_matrix();
    auto local_c = c.get_matrix();

    printf("A:\n");
    print_vector(local_a, a.shape()[0], a.shape()[1]);
    printf("B:\n");
    print_vector(local_b, b.shape()[0], b.shape()[1]);
    printf("C:\n");
    print_vector(local_c, c.shape()[0], c.shape()[1]);
  }

  BCL::print("Multiplying matrices...\n");
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();

  gemm(a, b, c);

  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  fflush(stdout);
  BCL::barrier();
  fflush(stdout);
  sleep(1);
  BCL::barrier();

  double n_gflops = num_gflops(c.shape()[0], c.shape()[1], a.shape()[1]);

  double gflops = n_gflops / duration;

  BCL::print("Done in %lf s (%lf GFLOPs)\n", duration, gflops);
  BCL::print("%lf TFLOPs/GPU (%lf%% of peak)\n",
             1e-3*(gflops / BCL::nprocs()), (100*(1e-3*(gflops / BCL::nprocs())) / 15.7));

  duration_issue = BCL::allreduce(duration_issue, std::plus<double>{});
  duration_sync = BCL::allreduce(duration_sync, std::plus<double>{});
  duration_compute = BCL::allreduce(duration_compute, std::plus<double>{});
  duration_barrier = BCL::allreduce(duration_barrier, std::plus<double>{});

  if (BCL::rank() == 0) {
    printf("duration_issue %lf\n", duration_issue / BCL::nprocs());
    printf("duration_sync %lf\n", duration_sync / BCL::nprocs());
    printf("duration_compute %lf\n", duration_compute / BCL::nprocs());
    printf("duration_barrier %lf\n", duration_barrier / BCL::nprocs());
  }

  if (BCL::rank() == 0 && test_result) {
    printf("Pulling matrices...\n");
    fflush(stdout);
    auto local_a = a.get_matrix();
    auto local_b = b.get_matrix();

    printf("Locally computing result...\n");
    fflush(stdout);
    std::vector<matmul_type> local_c(c.shape()[0]*c.shape()[1], 0);
    /*
    for (size_t i = 0; i < c.shape()[0]; i++) {
      for (size_t j = 0; j < c.shape()[1]; j++) {
        for (size_t k = 0; k < a.shape()[1]; k++) {
          local_c[i*c.shape()[1] + j] += local_a[i*a.shape()[1] + k] * local_b[k*b.shape()[1] + j];
        }
      }
    }
    */
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               c.shape()[0], c.shape()[1], a.shape()[1],
               1.0,
               local_a.data(), a.shape()[1],
               local_b.data(), b.shape()[1],
               1.0,
               local_c.data(), c.shape()[1]);

    auto gpu_matrix = c.get_matrix();

    matmul_type eps = 1e-5;
    for (size_t i = 0; i < c.shape()[0]; i++) {
      for (size_t j = 0; j < c.shape()[1]; j++) {
        size_t idx = i*c.shape()[1] + j;
        if (std::abs(local_c[idx] - gpu_matrix[idx]) > eps) {
          printf("(%lu, %lu) %f vs %f too big! (%f)\n",
                 i, j,
                 local_c[idx], gpu_matrix[idx], std::abs(local_c[idx] - gpu_matrix[idx]));
          // assert(false);
        }
      }
    }
    printf("OK!\n");
  }

  BCL::cuda::barrier();

  BCL::finalize();
  return 0;
}
