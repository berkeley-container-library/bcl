
#define __thrust_compiler_fence() __sync_synchronize()
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

// #include <bcl/containers/sequential/CSRMatrix.hpp>
#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/sequential/CSRMatrix.hpp>
#include <bcl/containers/experimental/cuda/sequential/CudaCSRMatrix.hpp>

#include "cusp_util.hpp"

#include <numeric>


int main(int argc, char** argv) {
	size_t num_args = 1;

  if (argc != num_args+1) {
    fprintf(stderr, "usage: ./analyze_matrix [input file]\n");
    fprintf(stderr, "Analyze matrix.\n");
    fprintf(stderr, "Automatically infers type from extensions (.mtx or .binary).\n");
    return 1;
  }

  std::string input_fname(argv[1]);

  auto format = BCL::matrix_io::detect_file_type(input_fname);
  assert(format == BCL::FileFormat::Binary);

  using index_type = int;

  fprintf(stderr, "Reading in \"%s\"\n", input_fname.c_str());
  BCL::CSRMatrix<float, index_type> a(input_fname);
  BCL::CSRMatrix<float, index_type> b(input_fname);

  fprintf(stderr, "Dimensions: %lu x %lu, NNZ: %lu\n", a.shape()[0], a.shape()[1], a.nnz());
  std::vector<double> iter_flops;
  std::vector<double> cfs;

  // for (size_t num_nodes : {1, 4, 9, 16, 36, 64, 81}) {
  for (size_t num_nodes : {64}) {
    size_t num_gpus = num_nodes*4;

    size_t p = num_gpus;
    size_t sqrtp = sqrt(p)+0.5;

    size_t m_s = (a.shape()[0] + sqrtp - 1) / sqrtp;
    size_t n_s = (a.shape()[1] + sqrtp - 1) / sqrtp;

    fprintf(stderr, "p=%lu, %lu x %lu grid, %lu x %lu tiles\n",
            p, sqrtp, sqrtp, m_s, n_s);

    for (size_t i = 0; i < sqrtp; i++) {
      for (size_t j = 0; j < sqrtp; j++) {
        size_t rank = i*sqrtp + j;
        fprintf(stderr, "  (%lu): computing C[%lu, %lu]\n", rank, i, j);
        for (size_t k = 0; k < sqrtp; k++) {
          auto local_a = a.get_slice_impl_(i*m_s, (i+1)*m_s,
                                           k*n_s, (k+1)*n_s);

          auto local_b = b.get_slice_impl_(k*m_s, (k+1)*m_s,
                                           j*n_s, (j+1)*n_s);

          std::vector <size_t> nnzs_a(m_s, 0);
          std::vector <size_t> nnzs_b(m_s, 0);

          for (size_t i_ = 0; i_ < m_s; i_++) {
            for (size_t j_ptr = local_a.rowptr_data()[i_]; j_ptr < local_a.rowptr_data()[i_+1]; j_ptr++) {
              size_t j_ = local_a.colind_data()[j_ptr];
              float value = local_a.values_data()[j_ptr];
              nnzs_a[j_]++;
            }
          }

          for (size_t i_ = 0; i_ < m_s; i_++) {
            for (size_t j_ptr = local_b.rowptr_data()[i_]; j_ptr < local_b.rowptr_data()[i_+1]; j_ptr++) {
              size_t j_ = local_b.colind_data()[j_ptr];
              float value = local_b.values_data()[j_ptr];
              nnzs_b[i_]++;
            }
          }

          size_t flops = 0;
          for (size_t i_ = 0; i_ < nnzs_a.size(); i_++) {
            flops += nnzs_a[i_]*nnzs_b[i_];
          }
          printf("Going to do %lu flops.\n", flops);

          auto cusp_a = BCL::cuda::get_cusp_view_sparse_cpu(local_a);
          auto cusp_b = BCL::cuda::get_cusp_view_sparse_cpu(local_b);

          cusp::coo_matrix<index_type, float, cusp::host_memory> cusp_c;

          cusp::multiply(cusp_a, cusp_b, cusp_c);
          size_t nnz = cusp_c.values.size();

          fprintf(stderr, "    (%lu) multiplying A[%lu, %lu] (%lu nnz)* B[%lu, %lu] (%lu nnz) -> C[%lu, %lu](%lu) (%lu nnz)\n",
                  rank, i, k, local_a.nnz(), k, j, local_b.nnz(),
                  i, j, k, nnz);
          fprintf(stderr, "    (%lu) computation was %lf GFlops, for a compression factor of %lf\n",
                  rank, 1e-9*double(flops), double(flops) / nnz);
          iter_flops.push_back(flops);
          cfs.push_back(double(flops) / nnz);

          double avg_flops = std::accumulate(iter_flops.begin(), iter_flops.end(), 0.0) / iter_flops.size();
          double avg_cf = std::accumulate(cfs.begin(), cfs.end(), 0.0) / cfs.size();
          fprintf(stderr, "      AVG CF, Flops, %lf, %lf\n", avg_cf, avg_flops);
        }
      }
    }
  }

/*
  std::vector<size_t> nnzs(10*10, 0);

  size_t m_s = (matrix.shape()[0] + 10 - 1) / 10;
  size_t n_s = (matrix.shape()[1] + 10 - 1) / 10;

  fprintf(stderr, "Dimensions: %lu x %lu, NNZ: %lu\n", matrix.shape()[0], matrix.shape()[1], matrix.nnz());

  for (size_t i = 0; i < matrix.m(); i++) {
    for (index_type j_ptr = matrix.rowptr_data()[i]; j_ptr < matrix.rowptr_data()[i+1]; j_ptr++) {
      index_type j = matrix.colind_data()[j_ptr];

      size_t tile_i = i / m_s;
      size_t tile_j = j / n_s;

      nnzs[tile_i*10 + tile_j]++;
    }
  }

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      // fprintf(stderr, "(%lu, %lu): %lu nnz\n", i, j, nnzs[i*10 + j]);
    }
  }

  size_t total = 0;
  size_t max = 0;
  for (const auto& nnz : nnzs) {
    total += nnz;
    max = std::max(max, nnz);
  }

  double avg = total / (10.0*10.0);
  double load_imbalance = max / avg;

  fprintf(stderr, "Avg NNZs %lf, max %lu, load imbalance %lf\n",
          avg, max, load_imbalance);
          */

	return 0;
}
