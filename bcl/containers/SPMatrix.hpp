
#pragma once

#include <bcl/bcl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <tuple>
#include <array>

#include <algorithm>
#include <numeric>

#include <bcl/containers/sequential/CSRMatrix.hpp>
#include <bcl/containers/sequential/SparseAccumulator.hpp>
#include <bcl/containers/detail/Blocking.hpp>

namespace BCL
{

// TODO: this is a little ridiculous.  Need something more flexible that
//       won't require partial specializations for more complex usage.
template <typename T, typename index_type, typename Allocator>
class future<CSRMatrix<T, index_type, Allocator>> {
public:
  using allocator_traits = std::allocator_traits<Allocator>;

  using IAllocator = typename allocator_traits:: template rebind_alloc<index_type>;
  size_t m_, n_, nnz_;
  future<std::vector<T, Allocator>> vals_;
  future<std::vector<index_type, IAllocator>> row_ptr_;
  future<std::vector<index_type, IAllocator>> col_ind_;

  future() = delete;
  future(const future&) = delete;

  future(future&&) = default;
  future& operator=(future&&) = default;

  future(size_t m, size_t n, size_t nnz,
         future<std::vector<T, Allocator>>&& vals,
         future<std::vector<index_type, IAllocator>>&& row_ptr,
         future<std::vector<index_type, IAllocator>>&& col_ind)
           : m_(m), n_(n), nnz_(nnz),
             vals_(std::move(vals)),
             row_ptr_(std::move(row_ptr)),
             col_ind_(std::move(col_ind)) {
  }

  CSRMatrix<T, index_type, Allocator> get() {
    return CSRMatrix<T, index_type, Allocator>(m_, n_, nnz_,
                                    std::move(vals_.get()),
                                    std::move(row_ptr_.get()),
                                    std::move(col_ind_.get()));
  }
};

template <typename T, typename index_type = int>
class SPMatrix {
public:

  using size_type = size_t;

  // NOTE: vals_[i], col_ind_[i] are of size nnz_[i];
  std::vector<BCL::GlobalPtr<T>> vals_;
  std::vector<BCL::GlobalPtr<index_type>> col_ind_;
  std::vector<size_type> nnzs_;

  // NOTE: row_ptr_[i] is of size tile_size_m_+1
  std::vector<BCL::GlobalPtr<index_type>> row_ptr_;

  // Size of *matrix* (in elements)
  size_type m_, n_;
  size_type nnz_;
  // Size of *processor* grid
  size_type pm_, pn_;
  // Size (in elements) of a *tile*
  size_type tile_size_m_, tile_size_n_;

  // Size of *tile* grid (in tiles)
  size_type grid_dim_m_, grid_dim_n_;

  std::unique_ptr<BCL::Team> team_ptr_;

  SPMatrix(const SPMatrix&) = delete;
  SPMatrix(SPMatrix&&) = default;

  template <typename TeamType>
  SPMatrix(const std::string& fname, Block&& blocking, const TeamType& team,
           FileFormat format = FileFormat::MatrixMarket) :
           team_ptr_(team.clone()) {
    init(fname, std::move(blocking), format);
  }

  SPMatrix(const std::string& fname, Block&& blocking = BCL::BlockOpt(),
           FileFormat format = FileFormat::MatrixMarket) :
          team_ptr_(new BCL::WorldTeam()) {
    init(fname, std::move(blocking), format);
  }

  void init(const std::string& fname, Block&& blocking, FileFormat format = FileFormat::MatrixMarket) {
    CSRMatrix<T, index_type> mat(fname, format);

    m_ = mat.m_;
    n_ = mat.n_;
    nnz_ = mat.nnz_;

    blocking.seed(mat.m_, mat.n_, BCL::nprocs(this->team()));

    pm_ = blocking.pgrid_shape()[0];
    pn_ = blocking.pgrid_shape()[1];

    tile_size_m_ = blocking.tile_shape()[0];
    tile_size_n_ = blocking.tile_shape()[1];

    if (pm_*pn_ > BCL::nprocs()) {
      throw std::runtime_error("DMatrix: tried to create a DMatrix with a too large p-grid.");
    }

    grid_dim_m_ = (m_ + tile_size_m_ - 1) / tile_size_m_;
    grid_dim_n_ = (n_ + tile_size_n_ - 1) / tile_size_n_;

    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        size_t lpi = i % pm_;
        size_t lpj = j % pn_;
        size_t proc = lpj + lpi*pn_;

        size_t nnz;
        BCL::GlobalPtr<T> vals;
        BCL::GlobalPtr<index_type> col_ind;
        BCL::GlobalPtr<index_type> row_ptr;
        if (BCL::rank() == proc) {
          auto slc = mat.get_slice_impl_(i*tile_size_m_, (i+1)*tile_size_m_,
                                         j*tile_size_n_, (j+1)*tile_size_n_);
          nnz = slc.nnz_;
          vals = BCL::alloc<T>(std::max<size_t>(1, slc.vals_.size()));
          col_ind = BCL::alloc<index_type>(std::max<size_t>(1, slc.col_ind_.size()));
          row_ptr = BCL::alloc<index_type>(std::max<size_t>(1, slc.row_ptr_.size()));

          if (vals == nullptr || col_ind == nullptr || row_ptr == nullptr) {
            throw std::runtime_error("SMatrix: ran out of memory!");
          }

          std::memcpy(vals.local(), slc.vals_.data(), sizeof(T)*slc.vals_.size());
          std::memcpy(col_ind.local(), slc.col_ind_.data(), sizeof(index_type)*slc.col_ind_.size());
          std::memcpy(row_ptr.local(), slc.row_ptr_.data(), sizeof(index_type)*slc.row_ptr_.size());
        }
        nnz = BCL::broadcast(nnz, proc);
        vals = BCL::broadcast(vals, proc);
        col_ind = BCL::broadcast(col_ind, proc);
        row_ptr = BCL::broadcast(row_ptr, proc);

        nnzs_.push_back(nnz);
        vals_.push_back(vals);
        col_ind_.push_back(col_ind);
        row_ptr_.push_back(row_ptr);
      }
    }
  }

  const BCL::Team& team() const {
    return *team_ptr_;
  }

  template <typename Allocator>
  void assign_tile(size_t i, size_t j,
                    const CSRMatrix<T, index_type, Allocator>& mat,
                    bool allow_redistribute = false) {
    if (!allow_redistribute) {
      assert(vals_[j + i*grid_shape()[1]].is_local());
    }

    BCL::GlobalPtr<T> vals = BCL::alloc<T>(std::max<size_t>(1, mat.vals_.size()));
    BCL::GlobalPtr<index_type> col_ind = BCL::alloc<index_type>(std::max<size_t>(1, mat.col_ind_.size()));
    BCL::GlobalPtr<index_type> row_ptr = BCL::alloc<index_type>(std::max<size_t>(1, mat.row_ptr_.size()));

    if (vals == nullptr || col_ind == nullptr || row_ptr == nullptr) {
      throw std::runtime_error("BCL::SpMatrix::assign_tile: ran out of memory");
    }

    std::copy(mat.vals_.begin(), mat.vals_.end(), vals.local());
    std::copy(mat.col_ind_.begin(), mat.col_ind_.end(), col_ind.local());
    std::copy(mat.row_ptr_.begin(), mat.row_ptr_.end(), row_ptr.local());

    std::swap(vals, vals_[j + i*grid_shape()[1]]);
    std::swap(col_ind, col_ind_[j + i*grid_shape()[1]]);
    std::swap(row_ptr, row_ptr_[j + i*grid_shape()[1]]);
    nnzs_[j + i*grid_shape()[1]] = mat.nnz_;

    // TODO: deal with this properly
    if (vals.is_local()) {
      BCL::dealloc<T>(vals);
      BCL::dealloc<index_type>(col_ind);
      BCL::dealloc<index_type>(row_ptr);
    }
  }

  void rebroadcast_tiles(const std::vector<size_t>& locales = {}) {
    if (locales.empty()) {
      for (size_t i = 0; i < vals_.size(); i++) {
        vals_[i] = BCL::broadcast(vals_[i], vals_[i].rank);
      }

      for (size_t i = 0; i < col_ind_.size(); i++) {
        col_ind_[i] = BCL::broadcast(col_ind_[i], col_ind_[i].rank);
      }

      for (size_t i = 0; i < row_ptr_.size(); i++) {
        row_ptr_[i] = BCL::broadcast(row_ptr_[i], row_ptr_[i].rank);
      }

      for (size_t i = 0; i < nnzs_.size(); i++) {
        nnzs_[i] = BCL::broadcast(nnzs_[i], vals_[i].rank);
      }
    } else {
      assert(locales.size() == vals_.size());

      for (size_t i = 0; i < vals_.size(); i++) {
        if (vals_[i].rank == BCL::rank() && locales[i] != BCL::rank()) {
          BCL::dealloc(vals_[i]);
        }
        vals_[i] = BCL::broadcast(vals_[i], locales[i]);
      }

      for (size_t i = 0; i < col_ind_.size(); i++) {
        if (col_ind_[i].rank == BCL::rank() && locales[i] != BCL::rank()) {
          BCL::dealloc(col_ind_[i]);
        }
        col_ind_[i] = BCL::broadcast(col_ind_[i], locales[i]);
      }

      for (size_t i = 0; i < row_ptr_.size(); i++) {
        if (row_ptr_[i].rank == BCL::rank() && locales[i] != BCL::rank()) {
          BCL::dealloc(row_ptr_[i]);
        }
        row_ptr_[i] = BCL::broadcast(row_ptr_[i], locales[i]);
      }

      for (size_t i = 0; i < nnzs_.size(); i++) {
        nnzs_[i] = BCL::broadcast(nnzs_[i], locales[i]);
      }
    }

    nnz_ = std::accumulate(nnzs_.begin(), nnzs_.end(), size_type(0));
  }

  std::array<size_t, 2> shape() const noexcept {
    return {m_, n_};
  }

  std::array<size_t, 2> grid_shape() const noexcept {
    return {grid_dim_m_, grid_dim_n_};
  }

  std::array<size_t, 2> tile_shape() const noexcept {
    return {tile_size_m_, tile_size_n_};
  }

  std::array<size_t, 2> pgrid_shape() const noexcept {
    return {pm_, pn_};
  }

  size_t nnz() const noexcept {
    return nnz_;
  }

  size_t tile_size() const noexcept {
    return tile_shape()[0] * tile_shape()[1];
  }


  auto complementary_block() const {
    return BCL::BlockCustom({tile_shape()[1], tile_shape()[0]},
                            {pgrid_shape()[1], pgrid_shape()[0]});
  }

  template <typename E>
  auto dry_product_block(const E& other) const {
    return BCL::BlockCustom({tile_shape()[0], other.tile_shape()[1]},
                            {pgrid_shape()[0], pgrid_shape()[1]});
  }

  std::array<size_t, 2> tile_shape(size_t i, size_t j) const noexcept {
    size_t m_size = std::min(tile_shape()[0], shape()[0] - i*tile_shape()[0]);
    size_t n_size = std::min(tile_shape()[1], shape()[1] - j*tile_shape()[1]);
    return {m_size, n_size};
  }

  size_type tile_nnz(size_type i, size_type j) const noexcept {
    return nnzs_[i*grid_shape()[1] + j];
  }

  using fetch_allocator = BCL::bcl_allocator<T>;

  size_t tile_locale(size_t i, size_t j) const {
    return vals_[j + i*grid_shape()[1]].rank;
  }

  template <typename U>
  bool intervals_overlap_(const std::pair<U, U>& a, const std::pair<U, U>& b) const noexcept {
    // XXX: is this correct? (I think so.)
    bool value = ((a.second > b.first) && (a.first <= b.first)) ||
                 ((a.first < b.second) && (a.first >= b.first));
                 /*
    std::cout << a.first << " -> " << a.second << "  overlaps? " <<
                 b.first << " -> " << b.second << " " << ((value) ? "true" : "false") << std::endl;
                 */
    return value;
  }

  CSRMatrix<T, index_type> get_slice_impl_(size_t mmin, size_t mmax,
                                                      size_t nmin, size_t nmax) const {
    mmax = std::min(mmax, shape()[0]);
    nmax = std::min(nmax, shape()[1]);

    assert(mmax >= mmin);
    assert(nmax >= nmin);

    using Allocator = BCL::bcl_allocator<T>;

    BCL::SparseHashAccumulator<T, index_type, Allocator> acc;

    std::vector<std::pair<decltype(arget_tile(0, 0)),
                          std::pair<index_type, index_type>
                          >
                > tiles;

    for (size_t grid_i = 0; grid_i < grid_shape()[0]; grid_i++) {
      if (intervals_overlap_(std::make_pair(mmin, mmax),
                             std::make_pair(grid_i*tile_shape()[0], (grid_i+1)*tile_shape()[0]))
          ) {
        for (size_t grid_j = 0; grid_j < grid_shape()[1]; grid_j++) {
          if (intervals_overlap_(std::make_pair(nmin, nmax),
                                 std::make_pair(grid_j*tile_shape()[1], (grid_j+1)*tile_shape()[1]))
              ) {
            // We need to grab and accumulate tile [grid_i, grid_j].
            auto tile = arget_tile(grid_i, grid_j);
            // Shift tile coordinates to *real* coordinates
            index_type offset_i = grid_i*tile_shape()[0];
            index_type offset_j = grid_j*tile_shape()[1];
            // Shift real coordinates to *slice* coordinates
            offset_i -= mmin;
            offset_j -= nmin;

            std::pair<index_type, index_type> offset{offset_i, offset_j};

            tiles.emplace_back(std::make_pair(std::move(tile),
                                              std::move(offset)));
          }
        }
      }
    }

    for (auto& tile : tiles) {
      acc.accumulate(tile.first.get(),
                     tile.second);
    }

    return acc.get_matrix(mmax - mmin, nmax - nmin);
  }

  template <typename Allocator = fetch_allocator>
  CSRMatrix<T, index_type, Allocator> get_tile(size_t i, size_t j) const {
    using allocator_traits = std::allocator_traits<Allocator>;
    using IAllocator = typename allocator_traits:: template rebind_alloc<index_type>;
    size_t m, n, nnz;
    m = tile_shape(i, j)[0];
    n = tile_shape(i, j)[1];

    nnz = tile_nnz(i, j);

    auto vals = BCL::arget<T, Allocator>(vals_[j + i*grid_shape()[1]], nnz);
    auto col_ind = BCL::arget<index_type, IAllocator>(col_ind_[j + i*grid_shape()[1]], nnz);
    auto row_ptr = BCL::arget<index_type, IAllocator>(row_ptr_[j + i*grid_shape()[1]], m+1);

    return CSRMatrix<T, index_type, Allocator> (m, n, nnz,
                                     std::move(vals.get()),
                                     std::move(row_ptr.get()),
                                     std::move(col_ind.get()));
  }

  template <typename Allocator = fetch_allocator>
  future<CSRMatrix<T, index_type, Allocator>> arget_tile(size_t i, size_t j) const {
    using allocator_traits = std::allocator_traits<Allocator>;
    using IAllocator = typename allocator_traits:: template rebind_alloc<index_type>;
    size_t m, n, nnz;
    m = tile_shape(i, j)[0];
    n = tile_shape(i, j)[1];

    nnz = tile_nnz(i, j);

    auto vals = BCL::arget<T, Allocator>(vals_[j + i*grid_shape()[1]], nnz);
    auto row_ptr = BCL::arget<index_type, IAllocator>(row_ptr_[j + i*grid_shape()[1]], m+1);
    auto col_ind = BCL::arget<index_type, IAllocator>(col_ind_[j + i*grid_shape()[1]], nnz);

    return future<CSRMatrix<T, index_type, Allocator>>(m, n, nnz,
                                            std::move(vals),
                                            std::move(row_ptr),
                                            std::move(col_ind));
  }

  // OPTIMIZE this later.
  template <typename Allocator = std::allocator<T>>
  auto get() const {
    using matrix_type = CSRMatrix<T, index_type, Allocator>;
    using future_type = BCL::future<matrix_type>;

    std::vector<std::vector<future_type>> local_tiles(grid_shape()[0]);

    BCL::SparseHashAccumulator<T, index_type> acc;

    for (size_t i = 0; i < grid_shape()[0]; i++) {
      local_tiles[i].reserve(grid_shape()[1]);
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        local_tiles[i].emplace_back(std::move(arget_tile<Allocator>(i, j)));
      }
    }

    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        auto local_tile = local_tiles[i][j].get();

        size_t offset_i = i*tile_shape()[0];
        size_t offset_j = j*tile_shape()[1];

        acc.accumulate(std::move(local_tile), {offset_i, offset_j});
      }
    }

    auto mat = acc.get_matrix(shape()[0], shape()[1]);
    return std::move(mat);
  }

  // Sanity check: count the number of nonzeros
  size_t count_nonzeros_() const {
    size_t num_nonzeros = 0;
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        auto my_tile = get_tile(i, j);
        for (size_t i_ = 0; i_ < my_tile.m_; i_++) {
          for (size_t j_ = my_tile.row_ptr_[i_]; j_ < my_tile.row_ptr_[i_+1]; j_++) {
            num_nonzeros++;
          }
        }
      }
    }
    return num_nonzeros;
  }

  void print_details() const {
    printf("%lu x %lu matrix.\n", shape()[0], shape()[1]);
    printf("  Split among %lu %lu x %lu tiles\n", grid_shape()[0]*grid_shape()[1],
           tile_shape()[0], tile_shape()[1]);
    printf("  Forming a %lu x %lu tile grid\n", grid_shape()[0], grid_shape()[1]);
    printf("  On a %lu x %lu processor grid\n", pgrid_shape()[0], pgrid_shape()[1]);
  }
};

}
