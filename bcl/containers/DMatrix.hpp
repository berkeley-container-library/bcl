// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/bcl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <list>
#include <memory>

#include <bcl/containers/algorithms/gemm.hpp>
#include <bcl/containers/detail/Blocking.hpp>
#include <bcl/containers/detail/index.hpp>

namespace BCL {

template <typename E>
class DExpr;

template <typename E>
class DTransposeView;

/// Distributed dense matrix data structure storing elements of type `T`.
template <typename T>
class DMatrix : public DExpr<DMatrix<T>> {
public:

  using matrix_dim = index<std::size_t>;

  using scalar_type = T;

  std::vector<BCL::GlobalPtr<T>> ptrs_;

  // Size of *matrix* (in elements)
  size_t m_, n_;
  // Size of *processor* grid
  size_t pm_, pn_;
  // Size (in elements) of a *tile*
  size_t tile_size_m_, tile_size_n_;

  // Size of *tile* grid (in tiles)
  size_t grid_dim_m_, grid_dim_n_;

  std::unique_ptr<BCL::Team> team_ptr_;

  DMatrix(const DMatrix&) = delete;
  DMatrix(DMatrix&&) = default;

  std::vector<BCL::UserTeam> column_teams_;
  std::vector<BCL::UserTeam> row_teams_;

  template <typename Blocking>
  void init(size_t m, size_t n, Blocking&& blocking) {
    blocking.seed(m, n, BCL::nprocs(team()));
    pm_ = blocking.pgrid_shape()[0];
    pn_ = blocking.pgrid_shape()[1];

    tile_size_m_ = blocking.tile_shape()[0];
    tile_size_n_ = blocking.tile_shape()[1];

    if (pm_*pn_ > BCL::nprocs(team())) {
      throw std::runtime_error("DMatrix: tried to create a DMatrix with a too large p-grid.");
    }

    if (tile_size_m_ == 0) {
      tile_size_m_ = (m + pm_ - 1) / pm_;
    }
    if (tile_size_n_ == 0) {
      tile_size_n_ = (n + pn_ - 1) / pn_;
    }

    grid_dim_m_ = (m + tile_shape()[0] - 1) / tile_shape()[0];
    grid_dim_n_ = (n + tile_shape()[1] - 1) / tile_shape()[1];

    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        size_t lpi = i % pm_;
        size_t lpj = j % pn_;
        size_t proc = lpj + lpi*pn_;
        BCL::GlobalPtr<T> ptr;
        if (team().in_team() && BCL::rank(team()) == proc) {
          ptr = BCL::alloc<T>(tile_size());
        }
        ptr = BCL::broadcast(ptr, team().to_world(proc));
        if (ptr == nullptr) {
          throw std::runtime_error("DMatrix: ran out of memory!");
        }
        ptrs_.push_back(ptr);
      }
    }
    BCL::barrier();
  }

  template <typename V>
  void print_vec(const std::vector<V>& vals) {
    std::cout << "{";
    for (const auto& val : vals) {
      std::cout << val << " ";
    }
    std::cout << "}" << std::endl;
  }

  void init_teams() {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      std::vector<size_t> row_procs;
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        row_procs.push_back(tile_ptr({i, j}).rank);
      }
      /*
      if (BCL::rank() == 0) {
        printf("Row team %lu: ", i);
        print_vec(row_procs);
      }
      */
      row_teams_.push_back(BCL::UserTeam(row_procs));
    }

    for (size_t j = 0; j < grid_shape()[1]; j++) {
      std::vector<size_t> column_procs;
      for (size_t i = 0; i < grid_shape()[0]; i++) {
        column_procs.push_back(tile_ptr({i, j}).rank);
      }
      /*
      if (BCL::rank() == 0) {
        printf("Column team %lu: ", j);
        print_vec(column_procs);
      }
      */
      column_teams_.push_back(BCL::UserTeam(column_procs));
    }
  }

  /// Construct a distributed dense matrix of dimension `dim[0]` x `dim[1]`.
  /// The optional argument `blocking` describes how the matrix should be distributed, and
  /// the optional argument `team` controls on which subset of processes the matrix is stored.
  template <typename Blocking = BCL::BlockSquare,
            typename TeamType = BCL::WorldTeam>
  DMatrix(matrix_dim dim, Blocking&& blocking = Blocking(),
          TeamType&& team = TeamType()) : m_(dim[0]), n_(dim[1]),
          team_ptr_(team.clone()) {
    init(dim[0], dim[1], std::forward<Blocking>(blocking));
  }

  const BCL::Team& team() const {
    return *team_ptr_;
  }

  BCL::GlobalPtr<T> tile_ptr(matrix_dim index) {
    return ptrs_[index[0]*grid_shape()[1] + index[1]];
  }

  BCL::GlobalPtr<const T> tile_ptr(matrix_dim index) const {
    return ptrs_[index[0]*grid_shape()[1] + index[1]];
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  auto arget_tile(matrix_dim index) const {
    return std::make_tuple(BCL::arget<T, Allocator>(tile_ptr(index), tile_size()), is_transpose());
  }

  template <typename Allocator>
  auto arput_tile(matrix_dim index, std::vector<T, Allocator>&& tile) {
    return BCL::arput(tile_ptr(index), std::move(tile));
  }

  std::vector<T> get_tile(matrix_dim index) const {
    std::vector<T> vals(tile_size());
    BCL::rget(tile_ptr(index), vals.data(), tile_size());
    return vals;
  }

  std::vector<T> get_tile_row(matrix_dim index, size_t row) const {
    std::vector<T> vals(tile_shape(index)[1]);
    BCL::rget(tile_ptr(index) + row*tile_shape()[1], vals.data(), vals.size());
    return vals;
  }

  std::vector<T> get_row(size_t i) const {
    std::vector<T> vals(shape()[1]);

    // Retrieve row `row` of tile `tile_i`
    size_t tile_i = i / tile_shape()[0];
    size_t row = i % tile_shape()[0];

    size_t row_length = tile_shape()[1];

    for (size_t tile_j = 0; tile_j < grid_shape()[1]; tile_j++) {
      auto remote_ptr = tile_ptr({tile_i, tile_j}) + row*tile_shape()[1];
      size_t copy_length = sizeof(T)*tile_shape({tile_i, tile_j})[1];
      BCL::memcpy(vals.data() + tile_shape()[1]*tile_j, remote_ptr, copy_length);
    }
    return vals;
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  auto arget_tile_row(size_t i, size_t j, size_t row) const {
    return BCL::arget<T, Allocator>(tile_ptr({i, j}) + row*tile_shape()[1], tile_shape({i, j})[1]);
  }

  /// Return a reference to element `index[0]`, `index[1]` of the matrix.
  GlobalRef<T> operator[](matrix_dim index) {
    size_t pi = index[0] / tile_shape()[0];
    size_t pj = index[1] / tile_shape()[1];
    size_t local_i = index[0] % tile_shape()[0];
    size_t local_j = index[1] % tile_shape()[1];

    size_t p = pi*grid_shape()[1] + pj;
    size_t local_idx = local_i*tile_shape()[1] + local_j;
    return *(ptrs_[p] + local_idx);
  }

  GlobalRef<const T> operator[](matrix_dim index) const {
    size_t pi = index[0] / tile_shape()[0];
    size_t pj = index[1] / tile_shape()[1];
    size_t local_i = index[0] % tile_shape()[0];
    size_t local_j = index[1] % tile_shape()[1];

    size_t p = pi*grid_shape()[1] + pj;
    size_t local_idx = local_i*tile_shape()[1] + local_j;
    return *(ptrs_[p] + local_idx);
  }

  template <typename U>
  DMatrix& operator=(const U& value) {
    for (size_t i = 0; i < ptrs_.size(); i++) {
      if (ptrs_[i].is_local()) {
        T* lptr = ptrs_[i].local();
        // TODO: removed std::execution::par_unseq
        //       due to old Intel compiler on Cori.
        std::transform(
                       lptr, lptr + tile_size(),
                       lptr,
                       [&](auto&& elem) {
                         return value;
                       });
      }
    }
    return *this;
  }

  auto complementary_block() const {
    return BCL::BlockCustom({tile_shape()[1], tile_shape()[0]},
                            {pgrid_shape()[1], pgrid_shape()[0]});
  }

  DMatrix<T> complementary(size_t m, size_t n) const {
    assert(m == shape()[1]);
    return DMatrix<T>(m, n,
                      BCL::BlockCustom({tile_shape()[1], tile_shape()[0]},
                      {pgrid_shape()[1], pgrid_shape()[0]}), *team_ptr_);
  }

  template <typename U>
  DMatrix<T> dry_product(const DExpr<U>& other) const {
    // Inner dimensions must match.
    assert(shape()[1] == other.shape()[0]);
    // Inner dimensions of the tiles we're multiplying must match.
    assert(grid_shape()[1] == other.grid_shape()[0]);
    assert(tile_shape()[1] == other.tile_shape()[0]);

    return DMatrix<T>(shape()[0], other.shape()[1],
                      BCL::BlockCustom({tile_shape()[0], other.tile_shape()[1]},
                      {pgrid_shape()[0], pgrid_shape()[1]}), *team_ptr_);
  }

  /// Multiply the matrix by another matrix or symbolic matrix view, returning
  /// the result as a new matrix.
  template <typename MatrixType>
  [[nodiscard]] auto dot(MatrixType&& other) const {
    return BCL::experimental::gemm_two_args_impl_(*this, std::forward<MatrixType>(other));
  }

  /// Apply the functor `fn` in place on each element of the matrix.
  template <typename Fn>
  DMatrix& apply_inplace(Fn&& fn) {
    for (size_t i = 0; i < ptrs_.size(); i++) {
      if (ptrs_[i].is_local()) {
        T* lptr = ptrs_[i].local();
        std::transform(
                       lptr, lptr + tile_size(),
                       lptr,
                       [&](auto&& elem) {
                         return fn(elem);
                       });
      }
    }
    return *this;
  }

  /// Return a symbolic *transpose view* of the matrix.
  DTransposeView<DMatrix<T>> transpose() {
    return DTransposeView<DMatrix<T>>(*this);
  }

  /// Return a new matrix equal to the current matrix with `fn` applied to each
  /// element.
  template <typename Fn>
  [[nodiscard]] DMatrix<T> apply(Fn&& fn) const {
    DMatrix<T> result({shape()[0], shape()[1]},
                      BCL::BlockCustom({tile_shape()[0], tile_shape()[1]},
                                       {pm(), pn()}), *team_ptr_);
    for (size_t i = 0; i < ptrs_.size(); i++) {
      if (ptrs_[i].is_local()) {
        T* lptr = ptrs_[i].local();
        T* rptr = result.ptrs_[i].local();
        std::transform(
                       lptr, lptr + tile_size(),
                       rptr,
                       [&](auto&& elem) {
                         return fn(elem);
                       });
      }
    }
    return result;
  }

  /// Create a new matrix equal to the result of the binary operation `fn` applied
  /// element-wise to the elements of the current matrix and matrix `other`.
  template <typename Fn>
  DMatrix<T> binary_op(const DMatrix<T>& other, Fn&& fn) const {
    assert(shape() == other.shape());
    assert(tile_shape() == other.tile_shape());
    assert(pm() == pn());

    DMatrix<T> result(shape()[0], shape()[1],
                      pm(), pn(),
                      tile_shape()[0], tile_shape()[1]);

    for (size_t i = 0; i < ptrs_.size(); i++) {
      if (ptrs_[i].is_local()) {
        T* aptr = ptrs_[i].local();
        T* bptr = other.ptrs_[i].local();
        T* cptr = result.ptrs_[i].local();
        for (size_t j = 0; j < tile_size(); j++) {
          cptr[j] = fn(aptr[j], bptr[j]);
        }
      }
    }
    return result;
  }

  /// Apply the binary operation `fn` elementwise to the current matrix and
  /// `other`, storing the result in the current matrix.
  template <typename Fn>
  DMatrix<T>& binary_op_inplace(const DMatrix<T>& other, Fn&& fn) {
    if (!team().in_team()) {
      return *this;
    }
    assert(shape() == other.shape());
    assert(tile_shape() == other.tile_shape());
    assert(pgrid_shape() == other.pgrid_shape());

    for (size_t i = 0; i < ptrs_.size(); i++) {
      if (ptrs_[i].is_local()) {
        T* aptr = ptrs_[i].local();
        T* bptr = other.ptrs_[i].local();
        for (size_t j = 0; j < tile_size(); j++) {
          aptr[j] = fn(aptr[j], bptr[j]);
        }
      }
    }
    return *this;
  }

  DMatrix<T> operator+(const DMatrix<T>& other) const {
    return binary_op(other, std::plus<T>{});
  }

  DMatrix<T> operator-(const DMatrix<T>& other) const {
    return binary_op(other, std::minus<T>{});
  }

  DMatrix<T> operator*(const DMatrix<T>& other) const {
    return binary_op(other, std::multiplies<T>{});
  }

  DMatrix<T> operator/(const DMatrix<T>& other) const {
    return binary_op(other, std::divides<T>{});
  }

  DMatrix<T>& operator+=(const DMatrix<T>& other) {
    return binary_op_inplace(other, std::plus<T>{});
  }

  DMatrix<T> copy() const {
    return apply([](auto a) { return a; });
  }

  static bool is_transpose() {
    return false;
  }

  /// Return the shape of the matrix
  matrix_dim shape() const noexcept {
    return {m_, n_};
  }

  matrix_dim grid_shape() const noexcept {
    return {grid_dim_m_, grid_dim_n_};
  }

  matrix_dim pgrid_shape() const noexcept {
    return {pm(), pn()};
  }

  matrix_dim tile_shape() const noexcept {
    return {tile_size_m_, tile_size_n_};
  }


  matrix_dim tile_shape(matrix_dim index) const noexcept {
    size_t m_size = std::min(tile_size_m_, m_ - index[0]*tile_size_m_);
    size_t n_size = std::min(tile_size_n_, n_ - index[1]*tile_size_n_);
    return {m_size, n_size};
  }

  size_t tile_size() const noexcept {
    return tile_shape()[0]*tile_shape()[1];
  }

  size_t tile_rank(matrix_dim index) const noexcept {
    return tile_ptr(index).rank;
  }

  size_t pm() const noexcept {
    return pm_;
  }

  size_t pn() const noexcept {
    return pn_;
  }

  // TODO: Should really do this by iterating
  // through submatrices...
  // What's a nice syntax for that?
  std::vector<T> get_matrix() const {
    std::vector<T> my_matrix(shape()[0]*shape()[1]);
    for (size_t i = 0; i < shape()[0]; i++) {
      for (size_t j = 0; j < shape()[1]; j++) {
        my_matrix[i*shape()[1] + j] = (*this)(i, j);
      }
    }
    return my_matrix;
  }

  /// Print the matrix to the terminal
  void print() const {
    BCL::barrier();
    if (BCL::rank() == 0) {
      for (size_t i = 0; i < shape()[0]; i++) {
        for (size_t j = 0; j < shape()[1]; j++) {
          std::cout << (*this)[{i, j}] << " ";
        }
        std::cout << std::endl;
      }
    }
    BCL::barrier();
  }

  void print_details() const {
    printf("%lu x %lu matrix.\n", shape()[0], shape()[1]);
    printf("  Split among %lu %lu x %lu tiles\n", grid_shape()[0]*grid_shape()[1],
           tile_shape()[0], tile_shape()[1]);
    printf("  Forming a %lu x %lu tile grid\n", grid_shape()[0], grid_shape()[1]);
    printf("  On a %lu x %lu processor grid\n", pgrid_shape()[0], pgrid_shape()[1]);
  }

  void print_info(bool print_pgrid = true) const {
    printf("=== MATRIX INFO ===\n");
    printf("%lu x %lu matrix\n", shape()[0], shape()[1]);
    printf("  * Consists of %lu x %lu tiles\n", tile_shape()[0], tile_shape()[1]);
    printf("  * Arranged in a %lu x %lu grid\n", grid_shape()[0], grid_shape()[1]);

    if (print_pgrid) {
      for (size_t i = 0; i < grid_shape()[0]; i++) {
        printf("   ");
        for (size_t j = 0; j < grid_shape()[1]; j++) {
          printf("%2lu ", tile_ptr({i, j}).rank);
        }
        printf("\n");
      }
    }
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  BCL::future<std::vector<T, Allocator>> arslice(matrix_dim row_slice,
                                                 matrix_dim col_slice) const {
    return arslice_impl_(row_slice[0], row_slice[1], col_slice[0], col_slice[1]);
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  std::vector<T, Allocator> slice(matrix_dim row_slice,
                                  matrix_dim col_slice) const {
    return slice_impl_(row_slice[0], row_slice[1], col_slice[0], col_slice[1]);
  }

  // Get slice of [mmin, mmax), [nmin, nmax)
  template <typename Allocator = BCL::bcl_allocator<T>>
  std::vector<T, Allocator> slice_impl_(size_t mmin, size_t mmax, size_t nmin, size_t nmax) const {
    mmax = std::min(mmax, shape()[0]);
    nmax = std::min(nmax, shape()[1]);

    assert(mmax >= mmin);
    assert(nmax >= nmin);

    std::vector<T, Allocator> result((mmax - mmin)*(nmax - nmin));

    std::vector<BCL::request> requests;

    for (size_t i = mmin; i < mmax; i++) {
      // printf("Looking at row %lu\n", i);
      for (size_t j = nmin; j < nmax; j += tile_shape()[1] - (j % tile_shape()[1])) {
        size_t size = std::min(std::min(tile_shape()[1] - (j % tile_shape()[1]), nmax - j), shape()[1] - j);
        // printf("  Need to slice segment %lu -> %lu\n", j, j+size);
        // goes to
        //
        size_t ri = i - mmin;
        size_t rj = j - nmin;
        // printf("  Need to copy this starting at (%lu, %lu)\n", ri, rj);

        T* rloc = &result[ri*(nmax - nmin) + rj];
        auto request = arget(&(*this)(i, j), rloc, size);
        requests.push_back(request);
      }
    }

    for (auto& request : requests) {
      request.wait();
    }

    return result;
  }

  T sum() const {
    T local_sum = 0;
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_rank({i, j}) == BCL::rank()) {
          const T* values = tile_ptr({i, j}).local();

          for (size_t i_ = 0; i_ < tile_shape({i, j})[0]; i_++) {
            for (size_t j_ = 0; j_ < tile_shape({i, j})[1]; j_++) {
              local_sum += values[i_*tile_shape()[1] + j_];
            }
          }
        }
      }
    }

    return BCL::allreduce(local_sum, std::plus<T>{});
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  BCL::future<std::vector<T, Allocator>> arslice_impl_(size_t mmin, size_t mmax, size_t nmin, size_t nmax) const {
    mmax = std::min(mmax, shape()[0]);
    nmax = std::min(nmax, shape()[1]);

    assert(mmax >= mmin);
    assert(nmax >= nmin);

    std::vector<T, Allocator> result((mmax - mmin)*(nmax - nmin));

    std::vector<BCL::request> requests;

    for (size_t i = mmin; i < mmax; i++) {
      // printf("Looking at row %lu\n", i);
      for (size_t j = nmin; j < nmax; j += tile_shape()[1] - (j % tile_shape()[1])) {
        size_t size = std::min(std::min(tile_shape()[1] - (j % tile_shape()[1]), nmax - j), shape()[1] - j);
        // printf("  Need to slice segment %lu -> %lu\n", j, j+size);
        // goes to
        //
        size_t ri = i - mmin;
        size_t rj = j - nmin;
        // printf("  Need to copy this starting at (%lu, %lu)\n", ri, rj);

        T* rloc = &result[ri*(nmax - nmin) + rj];
        auto request = arget(&(*this)(i, j), rloc, size);
        requests.emplace_back(request);
      }
    }

    return BCL::future<std::vector<T, Allocator>>(std::move(result), std::move(requests));
  }
};

template<typename T, typename U>
void fill_range(DMatrix<T>& mat, U bound) {
  for (size_t gi = 0; gi < mat.grid_shape()[0]; gi++) {
    for (size_t gj = 0; gj < mat.grid_shape()[1]; gj++) {
      if (mat.tile_ptr({gi, gj}).is_local()) {
        for (size_t i = 0; i < mat.tile_shape()[0]; i++) {
          for (size_t j = 0; j < mat.tile_shape()[1]; j++) {
            size_t i_ = gi * mat.tile_shape()[0] + i;
            size_t j_ = gj * mat.tile_shape()[1] + j;
            mat.tile_ptr({gi, gj}).local()[i*mat.tile_shape()[1] + j] = ((i_*mat.shape()[1] + j_) % bound);
          }
        }
      }
    }
  }
}

template <typename E>
class DExpr {
public:
  const E& get_() const {
    return static_cast<const E&>(*this);
  }

  auto shape() const {
    return get_().shape();
  }

  auto tile_shape() const {
    return get_().tile_shape();
  }

  auto tile_shape(size_t i, size_t j) const {
    return get_().tile_shape(i, j);
  }

  auto grid_shape() const {
    return get_().grid_shape();
  }

  auto pgrid_shape() const {
    return get_().pgrid_shape();
  }

  auto transpose() const {
    return get_().transpose();
  }

  auto is_transpose() const {
    return get_().is_transpose();
  }

  auto arget_tile(size_t i, size_t j) const {
    return get_().arget_tile(i, j);
  }
};

template <typename E>
class DTransposeView : public DExpr<DTransposeView<E>> {
public:
  using scalar_type = typename E::scalar_type;

  DTransposeView(const E& mat) : mat_(mat) {}

  index<std::size_t> shape() const {
    auto shp = mat_.shape();
    return {shp[1], shp[0]};
  }

  index<std::size_t> tile_shape() const {
    auto shp = mat_.tile_shape();
    return {shp[1], shp[0]};
  }

  index<std::size_t> grid_shape() const {
    auto shp = mat_.grid_shape();
    return {shp[1], shp[0]};
  }

  index<std::size_t> pgrid_shape() const {
    auto shp = mat_.pgrid_shape();
    return {shp[1], shp[0]};
  }

  index<std::size_t> tile_shape(index<std::size_t> idx) const {
    auto shp = mat_.tile_shape({idx[1], idx[0]});
    return {shp[1], shp[0]};
  }

  decltype(auto) transpose() const {
    return mat_;
  }

  auto is_transpose() const {
    return !mat_.is_transpose();
  }

  // TODO: must signify transpose
  auto arget_tile(index<std::size_t> index) const {
    auto tupl = mat_.arget_tile({index[1], index[0]});
    std::get<1>(tupl) = is_transpose();
    return tupl;
  }

  template <typename MatrixType>
  [[nodiscard]] auto dot(MatrixType&& other) const {
    return BCL::experimental::gemm_two_args_impl_(*this, std::forward<MatrixType>(other));
  }

private:
  const E& mat_;
};

}
