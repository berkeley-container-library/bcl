// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <bcl/containers/SPMatrix.hpp>
#include <bcl/containers/algorithms/spgemm.hpp>
#include <bcl/containers/HashMap.hpp>

namespace BCL {
std::unordered_map<std::size_t, void*> local_segments;

template <typename TeamType>

void set_local_segments(TeamType&& local_team) {
  BCL::print("Setting local segments...\n");
  for (auto&& rank : local_team.members_) {
    if (rank == BCL::rank()) {
      local_segments[BCL::rank()] = BCL::smem_base_ptr;
    } else {
      std::string shm_name = "BCL_Segment_" + std::to_string(rank);

      int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
      if (shm_fd == -1) {
        throw std::runtime_error("shm_open failed");
      }

      void* ptr = mmap(nullptr, BCL::shared_segment_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
      if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed in init");
      }
      local_segments[rank] = ptr;
    }
  }
  BCL::print("Set local segments.\n");
}

template <typename T>
T* cast_local_ptr(BCL::GlobalPtr<T> ptr) {
  auto iter = local_segments.find(ptr.rank);
  if (iter == local_segments.end()) {
    throw std::runtime_error("Tried to cast non-local GlobalPtr.");
  }
  return (T *) (((char*) (*iter).second) + ptr.ptr);
}

} // end BCL

#include <bcl/containers/algorithms/experimental_gemm.hpp>

#include "generate_spmat.hpp"

#include <chrono>

template <typename T, std::size_t N = 2048>
class static_vector {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  static_vector() = default;
  ~static_vector() = default;
  static_vector(const static_vector&) = default;
  static_vector(static_vector&&) = default;
  static_vector& operator=(const static_vector&) = default;
  static_vector& operator=(static_vector&&) = default;

  template <typename U, typename Allocator>
  operator std::vector<U, Allocator>() const {
    return std::vector<U, Allocator>(begin(), end());
  }

  static_vector(std::size_t n, T value = T{}) {
    for (size_t i = 0; i < n; i++) {
      data_[i] = value;
    }
    size_ = n;
  }

  size_type size() const noexcept {
    return size_;
  }

  size_type capacity() const noexcept {
    return N;
  }

  reference operator[](std::size_t pos) noexcept { return data_[pos]; }
  const_reference operator[](std::size_t pos) const noexcept { return data_[pos]; }

  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept { return data_ + size_; }

  void push_back(const T& value) {
    if (size()+1 > capacity()) { std::abort(); }
    else { data_[size_++] = value; }
  }

  void push_back(T&& value) {
    if (size()+1 > capacity()) { std::abort(); }
    else { data_[size_++] = std::move(value); }
  }

private:
  T data_[N];
  std::size_t size_ = 0;
};


std::pair<BCL::UserTeam, BCL::UserTeam>
get_node_teams() {
  assert(BCL::nprocs() < 2048);
  BCL::HashMap<std::string, static_vector<std::size_t>> map(BCL::nprocs()*2);
  std::string host = BCL::hostname();
  map.modify(host, [](auto procs) {
    procs.push_back(BCL::rank());
    return procs;
  });

  BCL::barrier();

  using value_type = decltype(map)::value_type;

  static_vector<std::size_t> lead_procs;
  static_vector<std::size_t> node_procs;
  for (auto iter = map.begin(); iter != map.end(); ++iter) {
    value_type entry = *iter;
    auto&& [hostname, procs] = entry;
    std::sort(procs.begin(), procs.end());
    lead_procs.push_back(procs[0]);
    if (host == hostname) {
      node_procs = procs;
    }
  }
  BCL::UserTeam lead_team(lead_procs);
  BCL::UserTeam node_team(node_procs);

/*
  size_t seg_size = BCL::nprocs()*sizeof(int);
  std::string shm_name = "BCL Node Shared " + std::to_string(BCL::rank());
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  ftruncate(shm_fd, seg_size);
  void* seg_ptr = mmap(nullptr, seg_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (seg_ptr == MAP_FAILED) {
    throw std::runtime_error("mmap failed on leader");
  }

  munmap(seg_ptr, seg_size);
  shm_unlink(shm_name.c_str());
  */

  return std::pair(std::move(lead_team), std::move(node_team));
}


/*
  if (BCL::rank() == 0) {
    printf("Diagonistics from rank 0:\n");
    for (auto iter = map.begin(); iter != map.end(); ++iter) {
      value_type entry = *iter;
      auto&& [hostname, procs] = entry;
      printf("Host \"%s\"\n", hostname.c_str());
      std::sort(procs.begin(), procs.end());
      for (auto&& proc : procs) {
        printf("  Rank %lu\n", proc);
      }
    }
    printf("Created lead_team:\n");
    printf("  Size: %lu\n", BCL::nprocs(lead_team));
    for (size_t i = 0; i < lead_team.members_.size(); i++) {
      printf("   R %lu\n", lead_team.members_[i]);
    }
  }
  */


/*
  BCL::print("Iterating through matrix...\n");

  if (BCL::rank() == 0) {
    printf("A: %lu x %lu\n",
           a.grid_shape()[0], a.grid_shape()[1]);

    for (size_t i = 0; i < a.grid_shape()[0]; i++) {
      for (size_t j = 0; j < a.grid_shape()[1]; j++) {
        printf("%lu", a.tile_locale(i, j));
      }
      printf("\n");
    }

    printf("B: %lu x %lu\n",
           b.grid_shape()[0], b.grid_shape()[1]);

    for (size_t i = 0; i < b.grid_shape()[0]; i++) {
      for (size_t j = 0; j < b.grid_shape()[1]; j++) {
        printf("%lu", b.tile_locale(i, j));
      }
      printf("\n");
    }

    printf("C: %lu x %lu\n",
           c.grid_shape()[0], c.grid_shape()[1]);

    for (size_t i = 0; i < c.grid_shape()[0]; i++) {
      for (size_t j = 0; j < c.grid_shape()[1]; j++) {
        printf("%lu", c.tile_locale(i, j));
      }
      printf("\n");
    }
  }
  */

namespace BCL {

template <typename T, typename I,
          typename LeadTeamType,
          typename NodeTeamType>
void batched_rowwise_gemm_node(const BCL::SPMatrix<T, I>& a,
                               const BCL::DMatrix<T>& b,
                               BCL::DMatrix<T>& c,
                               LeadTeamType&& lead_team,
                               NodeTeamType&& node_team) {
  // Because of local c opt.
  assert(c.grid_shape()[1] == 1);

  BCL::GlobalPtr<std::tuple<size_t, size_t, T>> indices = nullptr;
  std::tuple<size_t, size_t, T>* indices_ = nullptr;
  size_t nnz;

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    for (size_t k = 0; k < a.grid_shape()[1]; k++) {
      if (a.tile_locale(i, k) == BCL::rank()) {
        indices = BCL::alloc<std::tuple<size_t, size_t, T>>(a.tile_nnz(i, k));
        if (indices == nullptr) {
          throw std::runtime_error("Ran out of memory...");
        }
        indices_ = cast_local_ptr(indices);
        T* values = a.vals_[i*a.grid_shape()[1] + k].local();
        I* row_ptr = a.row_ptr_[i*a.grid_shape()[1] + k].local();
        I* col_ind = a.col_ind_[i*a.grid_shape()[1] + k].local();
        T* local_c = c.tile_ptr(i, 0).local();
        nnz = 0;
        for (size_t i_ = 0; i_ < a.tile_shape(i, k)[0]; i_++) {
          for (size_t j_ptr = row_ptr[i_]; j_ptr < row_ptr[i_+1]; j_ptr++) {
            size_t k_ = col_ind[j_ptr];

            size_t i__ = i_ + i*a.tile_shape()[0];
            size_t k__ = k_ + k*a.tile_shape()[1];

            auto value = values[j_ptr];

            indices[nnz++] = std::tuple(i_, k__, value);
          }
        }

        auto begin = std::chrono::high_resolution_clock::now();
        std::sort(indices.local(), indices.local() + nnz,
                  [](auto a, auto b) {
                    return std::get<1>(a) < std::get<1>(b);
                  });
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - begin).count();
        if (BCL::rank() == 0) {
          printf("Duration of sort %lfs\n", duration);
        }
      }
    }
  }

  indices = BCL::broadcast(indices, 0, node_team);
  nnz = BCL::broadcast(nnz, 0, node_team);

  indices_ = cast_local_ptr(indices);
  // printf("%lu indices: %p\n", BCL::rank(), indices);

  std::vector<T> row;
  size_t current_row = std::numeric_limits<size_t>::max();
  size_t slice_size = (nnz + BCL::nprocs(node_team) - 1) / BCL::nprocs(node_team);
  BCL::GlobalPtr<T> local_c = nullptr;
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (node_team.in_team(c.tile_locale(i, j))) {
        local_c = c.tile_ptr(i, j);
      }
    }
  }

  T* local_c_ = cast_local_ptr(local_c);
  for (size_t i = slice_size*BCL::rank(node_team); i < (BCL::rank(node_team)+1)*slice_size; i++) {
    std::tuple<size_t, size_t, T> v = indices_[i];
    auto&& [i_, k_, value] = v;

    if (k_ != current_row) {
      auto begin = std::chrono::high_resolution_clock::now();
      row = b.get_row(k_);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      BCL::row_comm += duration;
      BCL::num_fetches++;
      current_row = k_;
    }

    for (size_t j_ = 0; j_ < row.size(); j_++) {
      // local_c[i_*c.tile_shape()[1] + j_] += row[j_]*value;
      // BCL::fetch_and_op(local_c + i_*c.tile_shape()[1] + j_, row[j_]*value, BCL::plus<T>{});
      // float c_val = local_c_[i_*c.tile_shape()[1] + j_];
      // local_c_[i_*c.tile_shape()[1] + j_] = c_val + row[j_]*value;
      local_c_[i_*c.tile_shape()[1] + j_] += row[j_]*value;
    }
  }
}

} // end BCL

int main(int argc, char** argv) {
  // How big to make each process' shared segment, in MB.
  size_t segment_size = 3096;
  bool threaded = false;

  BCL::init(segment_size, threaded);

  size_t m, k, n;

  // parameters: [number of samples] [number of categories] [embedding size] [nnz_row] [alpha]

  if (argc >= 4) {
    m = std::atoi(argv[1]);
    k = std::atoi(argv[2]);
    n = std::atoi(argv[3]);
  } else {
    BCL::finalize();
    return 1;
  }

  size_t nnz_per_row = 100;
  double alpha = 0.0;

  if (argc >= 5) {
    nnz_per_row = std::atoi(argv[4]);
  }

  if (argc >= 6) {
    alpha = std::atof(argv[5]);
  }

  auto&& [lead_team, node_team] = get_node_teams();
  set_local_segments(node_team);

  using value_type = float;
  using index_type = long long int;

  BCL::print("Generating blocks...\n");

  auto blocks = BCL::block_matmul(m, n, k);

  srand48(BCL::rank());
  BCL::print("Generating matrix (%lu x %lu), alpha %lf, nnz_per_row %lu\n",
             m, k, alpha, nnz_per_row);
  auto a = BCL::generate_matrix<value_type, index_type>(m, k, nnz_per_row, alpha, BCL::NewBlockRow{}, lead_team);

  BCL::DMatrix<value_type> b({k, n}, BCL::NewBlockRow{}, lead_team);
  BCL::DMatrix<value_type> c({m, n}, BCL::NewBlockRow{}, lead_team);

  BCL::print("Generated A (%lu x %lu matrix) with %lu nnz\n",
             a.shape()[0], a.shape()[1], a.nnz());

  BCL::print("Multipyling by B (%lu x %lu dense matrix)\n",
             b.shape()[0], b.shape()[1]);

  BCL::print("To produce C (%lu x %lu dense matrix)\n",
             c.shape()[0], c.shape()[1]);

/*
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_details();
    printf("B:\n");
    b.print_details();
    printf("C:\n");
    c.print_details();
  }
  */

  size_t real_nnz = a.count_nonzeros_();

  BCL::print("Counted %lu real nonzeros\n", real_nnz);

  size_t number_of_samples = 5;

  std::vector<double> times;

  for (size_t i = 0; i < number_of_samples; i++) {
    BCL::num_fetches = 0;
    BCL::row_comm = 0;
    b = 1;
    c = 0;

    size_t cache_size = 128*1024*1024;

    BCL::barrier();
    auto begin = std::chrono::high_resolution_clock::now();
    // BCL::rowwise_gemm(a, b, c);
    // BCL::cached_rowwise_gemm(a, b, c, cache_size);
    BCL::batched_rowwise_gemm_node(a, b, c, lead_team, node_team);
    // BCL::batched_rowwise_gemm(a, b, c);
    BCL::barrier();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();

    BCL::barrier();

    BCL::print("Matrix Multiply took %lf s\n", duration);

    BCL::print("Comm/comp %lf / %lf\n", BCL::row_comm, duration - BCL::row_comm);

    BCL::print("Sum is %lf\n", c.sum());

    size_t bytes_fetched = BCL::num_fetches*n*sizeof(value_type);
    double gb_s = (1.0e-9*bytes_fetched) / BCL::row_comm;
    /*
    fprintf(stderr, "(%lu) %lf GB/s %lu bytes fetched from in %lf seconds\n",
            BCL::rank(), gb_s, bytes_fetched, BCL::row_comm);
    fflush(stderr);
    BCL::barrier();
    fflush(stderr);
    usleep(100);
    BCL::barrier();
    */
    times.push_back(duration);
  }

  BCL::barrier();

  std::sort(times.begin(), times.end());


  size_t total_lookups = a.nnz();
  size_t lookup_bytes = sizeof(value_type)*b.shape()[1];
  double gb = 1e-9*total_lookups*lookup_bytes;
  double gb_s = gb / times[times.size() / 2];

  size_t bytes_fetched = BCL::num_fetches*n*sizeof(value_type);
  size_t actual_lookup_bytes = BCL::allreduce<size_t>(bytes_fetched, std::plus<size_t>{});
  double actual_gb = 1e-9*actual_lookup_bytes;
  double actual_gb_s = actual_gb / BCL::row_comm;

  BCL::print("Matrix Multiply took %lf s (median)\n", times[times.size()/2]);
  BCL::print("%lu lookups of %lu bytes (%lf GB/s) (~%lu actual lookups for %lf GB/s [%lf / proc])\n",
             total_lookups, lookup_bytes, gb_s,
             actual_lookup_bytes / (n*sizeof(value_type)), actual_gb_s, actual_gb_s / BCL::nprocs());

  BCL::finalize();

  return 0;
}
