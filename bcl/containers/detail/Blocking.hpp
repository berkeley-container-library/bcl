#pragma once

#include <cassert>
#include <initializer_list>
#include <cmath>

namespace BCL {

// Factor n into 2 roughly equal factors
// n = pq, p >= q
std::vector<size_t> factor(size_t n) {
  size_t q = std::sqrt(n);

  while (q > 1 && n / q != static_cast<double>(n) / q) {
    q--;
  }
  size_t p = n / q;

  return {p, q};
}

struct Tile {
  static constexpr size_t div = 0;
};

struct Block {
  virtual std::vector<size_t> tile_shape() const = 0;
  virtual std::vector<size_t> pgrid_shape() const = 0;
  virtual void seed(size_t m, size_t n, size_t nprocs) = 0;
};

struct BlockRect : public Block {
  BlockRect(std::initializer_list<size_t> tile_shape) {
    assert(tile_shape.size() == 2);
    tile_shape_ = tile_shape;
  }

  BlockRect() {
    tile_shape_.resize(2, BCL::Tile::div);
  }

  void seed(size_t m, size_t n, size_t nprocs = BCL::nprocs()) {
    nprocs_ = nprocs;
    pgrid_shape_ = factor(nprocs_);
  }

  std::vector<size_t> tile_shape() const override {
    return tile_shape_;
  }

  std::vector<size_t> pgrid_shape() const override {
    return pgrid_shape_;
  }

  BlockRect(const BlockRect&) = default;

  std::vector<size_t> tile_shape_;
  std::vector<size_t> pgrid_shape_;
  size_t nprocs_;
};

struct BlockSquare : public Block {
  BlockSquare(std::initializer_list<size_t> tile_shape) {
    assert(tile_shape.size() == 2);
    tile_shape_ = tile_shape;
  }

  std::vector<size_t> tile_shape() const override {
    return tile_shape_;
  }

  std::vector<size_t> pgrid_shape() const override {
    return pgrid_shape_;
  }

  void seed(size_t m, size_t n, size_t nprocs = BCL::nprocs()) {
    nprocs_ = nprocs;
    // XXX: this line causes Intel compiler to crash...
    // pgrid_shape_ = {std::sqrt(nprocs_), std::sqrt(nprocs_)};
    pgrid_shape_.resize(2);
    pgrid_shape_[0] = std::sqrt(nprocs_);
    pgrid_shape_[1] = std::sqrt(nprocs_);
  }

  std::vector<size_t> tile_shape_;
  std::vector<size_t> pgrid_shape_;
  size_t nprocs_;
};

struct BlockRow : public Block {
  BlockRow(std::initializer_list<size_t> tile_shape) {
    assert(tile_shape.size() == 2);
    tile_shape_ = tile_shape;
  }

  BlockRow() {
    tile_shape_.resize(2, BCL::Tile::div);
  }

  void seed(size_t m, size_t n, size_t nprocs = BCL::nprocs()) {
    nprocs_ = nprocs;
    pgrid_shape_ = {nprocs_, 1};
  }

  std::vector<size_t> tile_shape() const override {
    return tile_shape_;
  }

  std::vector<size_t> pgrid_shape() const override {
    return pgrid_shape_;
  }

  std::vector<size_t> tile_shape_;
  std::vector<size_t> pgrid_shape_;
  size_t nprocs_;
};

struct BlockColumn : public Block {
  BlockColumn(std::initializer_list<size_t> tile_shape) {
    assert(tile_shape.size() == 2);
    tile_shape_ = tile_shape;
  }

  BlockColumn() {
    tile_shape_ = {BCL::Tile::div, BCL::Tile::div};
  }

  void seed(size_t m, size_t n, size_t nprocs = BCL::nprocs()) {
    nprocs_ = nprocs;
    pgrid_shape_ = {1, nprocs_};
  }

  std::vector<size_t> tile_shape() const override {
    return tile_shape_;
  }

  std::vector<size_t> pgrid_shape() const override {
    return pgrid_shape_;
  }

  std::vector<size_t> tile_shape_;
  std::vector<size_t> pgrid_shape_;
  size_t nprocs_;
};

struct BlockOpt : public Block {
  BlockOpt() {
    tile_shape_ = {BCL::Tile::div, BCL::Tile::div};
  }

  void seed(size_t m, size_t n, size_t nprocs = BCL::nprocs()) {
    nprocs_ = nprocs;
    shape_ = {m, n};
    pgrid_shape_ = {0, 0};

    pgrid_shape_[0] = std::sqrt((shape_[0] * nprocs_) / static_cast<double>(shape_[1]));
    pgrid_shape_[1] = std::sqrt((shape_[1] * nprocs_) / static_cast<double>(shape_[0]));

    pgrid_shape_[0] = std::max(size_t(1), pgrid_shape_[0]);
    pgrid_shape_[1] = std::max(size_t(1), pgrid_shape_[1]);

    if (pgrid_shape_[0] * pgrid_shape_[1] != nprocs_) {
      size_t lower = (pgrid_shape_[0] <= pgrid_shape_[1]) ? 0 : 1;
      size_t upper = !lower;
      while (pgrid_shape_[lower] > 1 && nprocs_ / pgrid_shape_[lower] !=
             static_cast<double>(nprocs_) / pgrid_shape_[lower]) {
        pgrid_shape_[lower]--;
      }
      pgrid_shape_[upper] = nprocs_ / pgrid_shape_[lower];
    }

    size_t n_ks = 8;
    size_t min_k = 1024;

    if (tile_shape_[0] == BCL::Tile::div) {
      tile_shape_[0] = (m + pgrid_shape_[0] - 1) / pgrid_shape_[0];
    }

    if (tile_shape_[1] == BCL::Tile::div) {
      tile_shape_[1] = (n + pgrid_shape_[1] - 1) / pgrid_shape_[1];

      if (pgrid_shape_[1] < 8) {
        for (size_t i = n_ks; i >= pgrid_shape_[1]; i--) {
          tile_shape_[1] = (n + i - 1) / (i);
          if (tile_shape_[1] >= min_k) {
            break;
          }
        }
      }
    }
  }

  std::vector<size_t> tile_shape() const override {
    return tile_shape_;
  }

  std::vector<size_t> pgrid_shape() const override {
    return pgrid_shape_;
  }

  std::vector<size_t> shape_;
  std::vector<size_t> tile_shape_;
  std::vector<size_t> pgrid_shape_;
  size_t nprocs_;
};

struct BlockCustom : public Block {
  BlockCustom(std::initializer_list<size_t> tile_shape, std::initializer_list<size_t> grid_shape) {
    assert(tile_shape.size() == 2);
    assert(grid_shape.size() == 2);
    tile_shape_ = tile_shape;
    grid_shape_ = grid_shape;
  }

  void seed(size_t m, size_t n, size_t nprocs = BCL::nprocs()) {
    nprocs_ = nprocs;

    if (tile_shape_[0] == BCL::Tile::div) {
      tile_shape_[0] = (m + grid_shape_[0] - 1) / grid_shape_[0];
    }
    if (tile_shape_[1] == BCL::Tile::div) {
      tile_shape_[1] = (m + grid_shape_[1] - 1) / grid_shape_[1];
    }
  }

  std::vector<size_t> tile_shape() const override {
    return tile_shape_;
  }

  std::vector<size_t> pgrid_shape() const override {
    return grid_shape_;
  }

  std::vector<size_t> tile_shape_;
  std::vector<size_t> grid_shape_;
  size_t nprocs_;
};

std::vector<BCL::BlockCustom> block_matmul(size_t m,
                                           size_t n,
                                           size_t k,
                                           BCL::Team&& team = BCL::WorldTeam()) {
  // Split blocks of C evenly.
  auto c_pgrid = factor(team.nprocs());
  BCL::BlockCustom c_block({BCL::Tile::div, BCL::Tile::div}, {c_pgrid[0], c_pgrid[1]});

  // TODO: something smarter here.
  size_t k_block;
  if (m*k >= k*n) {
    k_block = (team.nprocs() + c_pgrid[0] - 1) / c_pgrid[0];
  } else {
    k_block = (team.nprocs() + c_pgrid[1] - 1) / c_pgrid[1];
  }

  BCL::BlockCustom a_block({BCL::Tile::div, BCL::Tile::div}, {c_pgrid[0], k_block});
  BCL::BlockCustom b_block({BCL::Tile::div, BCL::Tile::div}, {k_block, c_pgrid[1]});

  return {a_block, b_block, c_block};
}

}
