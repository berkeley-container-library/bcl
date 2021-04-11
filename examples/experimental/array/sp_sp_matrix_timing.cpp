// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

// #include <CombBLAS/CombBLAS.h>
#include <cmath>
#include <bcl/bcl.hpp>
#include <bcl/containers/SPMatrix.hpp>

#include <mkl.h>

#include <bcl/core/detail/hash_functions.hpp>

double synct = 0;
double issue = 0;
double acc = 0;
double compute = 0;
double load = 0;
double end_acc = 0;
double rebroadcast = 0;


template <typename index_type, typename value_type>
void compare(const std::vector<std::pair<std::pair<index_type, index_type>, value_type>>& a,
             const std::vector<std::pair<std::pair<index_type, index_type>, value_type>>& b,
             value_type rtole = 1e-05, value_type atole=1e-08);

template <
          typename T,
          typename index_type,
          typename Mult = std::multiplies<T>,
          typename Plus = std::plus<T>
          >
void async_gemm(const BCL::SPMatrix<T, index_type>& a,
                 const BCL::SPMatrix<T, index_type>& b,
                       BCL::SPMatrix<T, index_type>& c);

template <typename T>
bool allclose(std::vector<T> a, std::vector<T> b, T rtole = 1e-03, T atole = 1e-05);

int main(int argc, char **argv) {
  BCL::init();

  if (argc < 3) {
    BCL::print("usage: ./sp_sp_matrix [matrix_file] [verify (true|false)]\n");
    BCL::finalize();
    return 0;
  }

  // Get arguments.
  std::string fname = std::string(argv[1]);
  bool verify = std::string(argv[2]) == "true";

  // Detect, print out matrix format.
  auto mat_format = BCL::matrix_io::detect_file_type(fname);
  std::string format_str = (mat_format == BCL::FileFormat::MatrixMarket) ? "mtx" : "binary";
  BCL::print("Main matrix is in format \".%s\"\n",
             format_str.c_str());

  // Grab dimensions of the matrix to be multiplied.
  size_t m, n, k;
  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  m = matrix_shape.shape[0];
  n = matrix_shape.shape[1];
  assert(m == n);
  k = m;

  // Compute optimal blocking for A, B, C.
  auto blocking = BCL::block_matmul(m, n, k);

  BCL::print("Reading in sparse matrix \"%s\"...\n", fname.c_str());
  auto begin = std::chrono::high_resolution_clock::now();

  BCL::print("Reading in A!!\n");
  BCL::SPMatrix<float, MKL_INT> a(fname, std::move(blocking[0]));
  BCL::print("Reading in B!!\n");
  BCL::SPMatrix<float, MKL_INT> b(fname, std::move(blocking[1]));

  BCL::print("Reading in C!!\n");
  BCL::SPMatrix<float, MKL_INT> c(a.shape()[0], b.shape()[1], std::move(blocking[2]));

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  BCL::print("Read sparse matrix in %lfs.\n", duration);

  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_details();
    printf("B:\n");
    b.print_details();
    printf("C:\n");
    c.print_details();
  }

  BCL::barrier();

  BCL::print("A has %lu nnz\n", a.nnz());
  BCL::print("B has %lu nnz\n", b.nnz());

  BCL::print("Doing distributed GEMM...\n");
  BCL::print("Before GEMM, have %lu nnz\n", c.nnz());
  BCL::barrier();
  begin = std::chrono::high_resolution_clock::now();
  async_gemm(a, b, c);
  end = std::chrono::high_resolution_clock::now();
  double d_duration = std::chrono::duration<double>(end - begin).count();
  BCL::print("Done in %lfs\n", d_duration);

  BCL::barrier();
  printf("(%lu) sync %lf, issue %lf acc %lf compute %lf load %lf end_acc %lf rebroadcast %lf\n",
         BCL::rank(), synct, issue, acc, compute, load, end_acc, rebroadcast);
  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  bool local_compare = verify;
  bool mkl_check = verify;

  if (BCL::rank() == 0 && (local_compare || mkl_check)) {
    BCL::CSRMatrix<float, MKL_INT> a_local(fname);
    BCL::CSRMatrix<float, MKL_INT> b_local(fname);

    printf("Multiplying locally...\n");
    fflush(stdout);
    auto begin = std::chrono::high_resolution_clock::now();
    auto c_local = a_local.dot(b_local);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();

    printf("%lf s running MKL locally.\n", duration);
    fflush(stdout);

    if (mkl_check) {
      printf("Retrieving remote matrix...\n");
      fflush(stdout);
      auto c_dist = c.get();

      printf("NNZ %lu local %lu remote\n", c_local.nnz_, c_dist.nnz_);
      printf("Getting COOs...\n");
      fflush(stdout);
      auto cl_coo = c_local.get_coo();
      auto cd_coo = c_dist.get_coo();

      printf("Comparing...\n");
      fflush(stdout);
      compare(cl_coo, cd_coo);
    }
  }

  BCL::finalize();
}

#include <unordered_map>

template <
          typename T,
          typename index_type,
          typename Mult = std::multiplies<T>,
          typename Plus = std::plus<T>
          >
void async_gemm(const BCL::SPMatrix<T, index_type>& a,
                 const BCL::SPMatrix<T, index_type>& b,
                       BCL::SPMatrix<T, index_type>& c) {
  // accumulated C's: a map of grid coordinates to sparse
  //                  matrices (for now, also maps)
  //

  std::unordered_map<
                     std::pair<size_t, size_t>,
                     // BCL::SparseSPAAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     BCL::SparseHashAccumulator<T, index_type, Plus, BCL::bcl_allocator<T>>,
                     // BCL::SparseHeapAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     // BCL::CombBLASAccumulator<T, index_type>,
                     // BCL::EagerMKLAccumulator<T, index_type>,
                     BCL::djb2_hash<std::pair<size_t, size_t>>
                     >
                     accumulated_cs;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {

        size_t k_offset = j % a.grid_shape()[1];

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile(i, k_offset % a.grid_shape()[1]);
        auto buf_b = b.arget_tile(k_offset % a.grid_shape()[1], j);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - begin).count();
        issue += duration;

        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          begin = std::chrono::high_resolution_clock::now();
          auto my_a = buf_a.get();
          auto my_b = buf_b.get();
          end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration<double>(end - begin).count();
          synct += duration;

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile(i, (k+1) % a.grid_shape()[1]);
            buf_b = b.arget_tile((k+1) % a.grid_shape()[1], j);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end - begin).count();
            issue += duration;
          }

          begin = std::chrono::high_resolution_clock::now();
          auto c = my_a. template dot<Mult, Plus>(my_b);
          end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration<double>(end - begin).count();
          compute += duration;

          begin = std::chrono::high_resolution_clock::now();
          accumulated_cs[{i, j}].accumulate(std::move(c));
          end = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration<double>(end - begin).count();
          acc += duration;
        }
      }
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
  BCL::barrier();
  BCL::print("Done with compute, accumulating...\n");
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  load += duration;

  begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {
        /*
        auto c_local = c.get_tile(i, j);
        accumulated_cs[{i, j}].accumulate(c_local);
        */

        auto cmatrix = accumulated_cs[{i, j}].get_matrix(c.tile_shape(i, j)[0], c.tile_shape(i, j)[1]);

        c.assign_tile(i, j, cmatrix);
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  end_acc += duration;
  begin = std::chrono::high_resolution_clock::now();
  BCL::barrier();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  load += duration;
  begin = std::chrono::high_resolution_clock::now();
  c.rebroadcast_tiles();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  rebroadcast += duration;
}

template <typename T>
bool allclose(std::vector<T> a, std::vector<T> b, T rtole, T atole) {
  assert(a.size() == b.size());

  T total_diff = 0;
  T max_diff = 0;
  size_t num_off = 0;
  size_t num_zeros = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > (atole + rtole * std::abs(b[i]))) {
      std::cout << a[i] << " != " << b[i] << std::endl;
      if (std::abs(b[i]) < 1.0e-5) {
        num_zeros++;
      }
      // printf("%d off by %lf\n", i, std::abs(a[i] - b[i]));
      num_off++;
      max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
      total_diff += std::abs(a[i] - b[i]);
    }
  }

  if (num_off > 0) {
    printf("not allclose: {%f,%f,%f,%lu,%lu} {max,avg,total,num_off,num_zeros}\n",
           max_diff, total_diff / num_off, total_diff, num_off, num_zeros);
    return false;
  } else {
    return true;
  }
}

template <typename index_type, typename value_type>
void compare(const std::vector<std::pair<std::pair<index_type, index_type>, value_type>>& a,
             const std::vector<std::pair<std::pair<index_type, index_type>, value_type>>& b,
             value_type rtole, value_type atole) {

  if (a.size() != b.size()) {
    printf("Matrix A(%lu nnz) != B(%lu nnz)\n", a.size(), b.size());
    fflush(stdout);
  }
  assert(a.size() == b.size());

  size_t incorrect_indx = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::get<0>(a[i]) != std::get<0>(b[i])) {
      incorrect_indx++;
      // std::cout << std::get<0>(std::get<0>(a[i])) << " != " << std::get<1>(std::get<0>(a[i])) << std::endl;
    }
  }

  size_t incorrect_val = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(std::get<1>(a[i]) - std::get<1>(b[i])) > (atole + rtole * std::abs(std::get<1>(b[i])))) {
      std::cout << "  " << std::get<1>(a[i]) << " != " << std::get<1>(b[i]) << std::endl;
      incorrect_val++;
    }
  }
  if (incorrect_val == 0 && incorrect_indx == 0) {
    printf("OK!\n");
  } else {
    printf("FAIL\n");
    printf("%lu incorrect indices, %lu incorrect values\n",
           incorrect_indx, incorrect_val);
  }
}
