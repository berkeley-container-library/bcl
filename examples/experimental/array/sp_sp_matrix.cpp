// #include <CombBLAS/CombBLAS.h>
#include <cmath>
#include <bcl/bcl.hpp>
#include <bcl/containers/SPMatrix.hpp>

template <typename index_type, typename value_type>
void compare(const std::vector<std::pair<std::pair<index_type, index_type>, value_type>>& a,
             const std::vector<std::pair<std::pair<index_type, index_type>, value_type>>& b,
             value_type rtole = 1e-05, value_type atole=1e-08);

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

  BCL::SPMatrix<float, MKL_INT> a(fname, std::move(blocking[0]));
  BCL::SPMatrix<float, MKL_INT> b(fname, std::move(blocking[1]));

  BCL::SPMatrix<float, MKL_INT> c(a.shape()[0], b.shape()[1], std::move(blocking[2]));

  BCL::gemm(a, b, c);

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
