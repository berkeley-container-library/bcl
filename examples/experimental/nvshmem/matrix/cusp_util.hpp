#pragma once

#include <chrono>

namespace BCL {

namespace cuda {

void test_cusp() {
  // initialize matrix
  cusp::csr_matrix<int, int, cusp::device_memory> A;
  cusp::io::read_matrix_market_file(A, "./chesapeake_general.mtx");

  cusp::constant_functor<float> initialize(0);
  thrust::multiplies<float> combine;
  thrust::plus<float>       reduce;

  size_t k = 128;

  cusp::array2d<int, cusp::device_memory> B(A.num_cols, k);

  // allocate output matrix
  cusp::array2d<int, cusp::device_memory> C(A.num_rows, k);

  auto begin = std::chrono::high_resolution_clock::now();
  // compute y = A * x
  cusp::multiply(A, B, C, initialize, combine, reduce);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  printf("Multiply took %lf seconds.\n", duration);
}

template <typename MatrixType>
auto get_cusp_view_dense(MatrixType& x) {
  using T = typename MatrixType::value_type;
  thrust::device_ptr<T> begin(x.data());
  thrust::device_ptr<T> end(x.data()+x.size());
  auto values_view = cusp::make_array1d_view(begin, end);

  return cusp::make_array2d_view(x.shape()[0], x.shape()[1], x.ld(),
                                 values_view, cusp::row_major());
}

template <typename MatrixType>
auto get_cusp_view_sparse(MatrixType& x) {
  using value_type = typename MatrixType::value_type;
  using index_type = typename MatrixType::index_type;
  thrust::device_ptr<value_type> values_begin(x.values_data());
  thrust::device_ptr<value_type> values_end(x.values_data()+x.nnz());
  thrust::device_ptr<index_type> rowptr_begin(x.rowptr_data());
  thrust::device_ptr<index_type> rowptr_end(x.rowptr_data()+x.shape()[0]+1);
  thrust::device_ptr<index_type> colind_begin(x.colind_data());
  thrust::device_ptr<index_type> colind_end(x.colind_data()+x.nnz());

  auto values_view = cusp::make_array1d_view(values_begin, values_end);
  auto rowptr_view = cusp::make_array1d_view(rowptr_begin, rowptr_end);
  auto colind_view = cusp::make_array1d_view(colind_begin, colind_end);

  return cusp::csr_matrix_view<decltype(rowptr_view),
                               decltype(colind_view),
                               decltype(values_view)>
                        (x.shape()[0], x.shape()[1], x.nnz(),
                         rowptr_view, colind_view, values_view);
}


template <typename AMatrixType, typename BMatrixType, typename CMatrixType>
void spmm_cusp(AMatrixType& a,
               BMatrixType& b,
               CMatrixType& c)
{
  using Allocator = typename AMatrixType::allocator_type;
  if (a.nnz() == 0){ 
    return;
  }

  auto a_view = get_cusp_view_sparse(a);
  auto b_view = get_cusp_view_dense(b);
  auto c_view = get_cusp_view_dense(c);

  cusp::multiply(a_view, b_view, c_view);
  cudaDeviceSynchronize();
}

template <typename AMatrixType, typename BMatrixType, typename CMatrixType>
void spgemm_cusp(AMatrixType& a,
                 BMatrixType& b,
                 CMatrixType& c)
{
  using Allocator = typename AMatrixType::allocator_type;
  if (a.nnz() == 0 || b.nnz() == 0){ 
    return;
  }

  auto a_view = get_cusp_view_sparse(a);
  auto b_view = get_cusp_view_sparse(b);

  cusp::multiply(a_view, b_view, c);
  cudaDeviceSynchronize();
}

template <typename T, typename I, typename MatrixType>
BCL::cuda::CudaCSRMatrixView<T, I> get_view(MatrixType& x) {
  auto* values_data = thrust::raw_pointer_cast(&x.values[0]);
  auto* rowptr_data = thrust::raw_pointer_cast(&x.row_offsets[0]);
  auto* colind_data = thrust::raw_pointer_cast(&x.column_indices[0]);

  return BCL::cuda::CudaCSRMatrixView<T, I>(x.num_rows, x.num_cols, x.num_entries,
                                            values_data, rowptr_data, colind_data);
}


template <typename MatrixType>
auto get_cusp_view_sparse_cpu(MatrixType& x) {
  using value_type = typename MatrixType::value_type;
  using index_type = typename MatrixType::index_type;
  using viterator_type = value_type*;
  using iiterator_type = index_type*;
  viterator_type values_begin(x.values_data());
  viterator_type values_end(x.values_data()+x.nnz());
  iiterator_type rowptr_begin(x.rowptr_data());
  iiterator_type rowptr_end(x.rowptr_data()+x.shape()[0]+1);
  iiterator_type colind_begin(x.colind_data());
  iiterator_type colind_end(x.colind_data()+x.nnz());

  auto values_view = cusp::make_array1d_view(values_begin, values_end);
  auto rowptr_view = cusp::make_array1d_view(rowptr_begin, rowptr_end);
  auto colind_view = cusp::make_array1d_view(colind_begin, colind_end);

  return cusp::csr_matrix_view<decltype(rowptr_view),
                               decltype(colind_view),
                               decltype(values_view)>
                        (x.shape()[0], x.shape()[1], x.nnz(),
                         rowptr_view, colind_view, values_view);
}

} // end cuda

} // end BCL