
#pragma once

#include <cblas.h>

namespace BCL {
namespace experimental {

void cblas_gemm_wrapper_(const CBLAS_ORDER layout,
                         const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                         const int m, const int n, const int k,
                         const float alpha, const float* a, const int lda,
                         const float* b, const int ldb, const float beta,
                         float* c, const int ldc) {
  cblas_sgemm(layout, transa, transb,
              m, n, k,
              alpha, a, lda,
              b, ldb, beta,
              c, ldc);
}

void cblas_gemm_wrapper_(const CBLAS_ORDER layout,
                         const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                         const int m, const int n, const int k,
                         const double alpha, const double* a, const int lda,
                         const double* b, const int ldb, const double beta,
                         double* c, const int ldc) {
  cblas_dgemm(layout, transa, transb,
              m, n, k,
              alpha, a, lda,
              b, ldb, beta,
              c, ldc);
}

template <typename T>
std::vector<T> cblas_test(const std::vector<T>& a, const std::vector<T>& b,
                          size_t M, size_t N, size_t K) {
  std::vector<T> c(M*N);
  cblas_gemm_wrapper_(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      M, N, K,
                      1.0, a.data(), K,
                      b.data(), N, 1.0,
                      c.data(), N);
  return c;
}

}
}
