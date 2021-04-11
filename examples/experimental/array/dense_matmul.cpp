// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/containers/DMatrix.hpp>

double num_gflops(size_t M, size_t N, size_t K);

template <typename T>
bool allclose(std::vector<T> a, std::vector<T> b, T rtole = 1e-05, T atole = 1e-08);

int main(int argc, char** argv) {
  BCL::init();

  using T = float;
  size_t m = 532;
  size_t n = 123;
  size_t k = 723;
  BCL::DMatrix<T> a(m, k, BCL::BlockRect({1024, 1024}));
  auto b = a.complementary(k, n);
  auto c = a.dry_product(b);

  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_details();
    printf("B:\n");
    b.print_details();
    printf("C:\n");
    c.print_details();
  }

  BCL::fill_range(a, 71);
  BCL::fill_range(b, 121);
  c = 0;

  auto begin = std::chrono::high_resolution_clock::now();
  BCL::barrier();
  BCL::experimental::gemm(a, b, c);
  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();
  double gflops = num_gflops(m, n, k);

  double perf = gflops / duration;

  BCL::print("DGEMM - %lf GFLOPs (elapsed in %lf s)\n", perf, duration);

  constexpr bool check = true;

  if (BCL::rank() == 0 && check) {
    std::vector<T> local_c = c.get_matrix();

    auto local_a = a.get_matrix();
    auto local_b = b.get_matrix();
    auto begin_cblas = std::chrono::high_resolution_clock::now();
    std::vector<T> cblas_c = BCL::experimental::cblas_test(local_a, local_b,
                                                           c.shape()[0], c.shape()[1], a.shape()[1]);
    auto end_cblas = std::chrono::high_resolution_clock::now();

    double cblas_duration = std::chrono::duration<double>(end_cblas - begin_cblas).count();

    printf("CBLAS got %lf GFLOPs (elapsed %lf)\n",
           num_gflops(c.shape()[0], c.shape()[1], a.shape()[1]) / cblas_duration, cblas_duration);

    bool correct = allclose<T>(local_c, cblas_c);

    if (correct) {
      printf("W00t! We're correct.\n");
    }
  }

  BCL::finalize();
  return 0;
}

template <typename T>
bool allclose(std::vector<T> a, std::vector<T> b, T rtole, T atole) {
  assert(a.size() == b.size());

  T total_diff = 0.0f;
  T max_diff = 0.0f;
  size_t num_off = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > atole + rtole * std::abs(b[i])) {
      num_off++;
      max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
      total_diff += std::abs(a[i] - b[i]);
    }
  }

  if (num_off > 0) {
    printf("not allclose: {%f,%f,%f,%lu} {max,avg,total,num_off}\n",
           max_diff, total_diff / num_off, total_diff, num_off);
    return false;
  } else {
    return true;
  }
}

double num_gflops(size_t M, size_t N, size_t K) {
  return 1e-9 * (2*M*N*K + 3*M*N);
}
