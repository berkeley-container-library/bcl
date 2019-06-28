#include <cmath>
#include <bcl/bcl.hpp>
#include <bcl/containers/SPMatrix.hpp>
#include <bcl/containers/ManyToManyDistributor.hpp>

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

template <typename T, typename index_type>
void async_gemm(const BCL::SPMatrix<T, index_type>& a,
                 const BCL::SPMatrix<T, index_type>& b,
                       BCL::SPMatrix<T, index_type>& c);

template <typename T>
bool allclose(std::vector<T> a, std::vector<T> b, T rtole = 1e-03, T atole = 1e-05);

auto detect_file_type(const std::string& fname) {
  size_t suffix_spot = 0;

  for (int64_t i = fname.size()-1; i >= 0; i--) {
    if (fname[i] == '.') {
      suffix_spot = i;
    }
  }

  std::string suffix = fname.substr(suffix_spot);

  if (suffix == ".mtx") {
    return BCL::FileFormat::MatrixMarket;
  } else if (suffix == ".binary") {
    return BCL::FileFormat::Binary;
  } else {
    assert(false);
    return BCL::FileFormat::Unknown;
  }
}

int main(int argc, char **argv) {
  BCL::init(8192);

  if (argc < 3) {
    BCL::print("usage: ./sp_sp_matrix [matrix_file] [zero_file] [verify (true|false)]\n");
    BCL::finalize();
    return 0;
  }

  std::string fname = std::string(argv[1]);
  std::string zero_fname = std::string(argv[2]);
  bool verify = std::string(argv[3]) == "true";

  auto mat_format = detect_file_type(fname);
  auto zero_format = detect_file_type(zero_fname);

  std::string format_str = (mat_format == BCL::FileFormat::MatrixMarket) ? "mtx" : "binary";
  std::string zero_format_str = (zero_format == BCL::FileFormat::MatrixMarket) ? "mtx" : "binary";

  BCL::print("Main matrix is in format \".%s\", zero matrix is in format \".%s\"\n",
             format_str.c_str(), zero_format_str.c_str());

  /*
  std::string dir = "/global/cscratch1/sd/brock/sparse-data/nlpkkt160/";
  // std::string fname = "nlpkkt160_general.binary";
  std::string fname = "nlpkkt160_general.mtx";
  std::string zero_fname = "zero_" + fname;
  // std::string zero_fname = "zero_nlpkkt160_general.mtx";
  */

  size_t m, n, k;

  {
    BCL::CSRMatrix<float, MKL_INT> a(zero_fname, zero_format);
    m = a.shape()[0];
    n = a.shape()[1];
    assert(m == n);
    k = m;
  }

  auto blocking = BCL::block_matmul(m, n, k);

  BCL::print("Reading in sparse matrix \"%s\"...\n", fname.c_str());
  auto begin = std::chrono::high_resolution_clock::now();

  size_t block_factor = 16;
  size_t m_block = (m + block_factor - 1) / block_factor;
  size_t n_block = (n + block_factor - 1) / block_factor;

  BCL::SPMatrix<float, MKL_INT> a(fname, std::move(blocking[0]), mat_format);
  // BCL::SPMatrix<float, MKL_INT> a(fname, BCL::BlockRect({m_block, n_block}), mat_format);

  BCL::SPMatrix<float, MKL_INT> b(fname, std::move(blocking[1]), mat_format);
  // BCL::SPMatrix<float, MKL_INT> b(fname, a.complementary_block(), mat_format);

  BCL::SPMatrix<float, MKL_INT> c(zero_fname, std::move(blocking[2]), zero_format);
  // BCL::SPMatrix<float, MKL_INT> c(zero_fname, a.dry_product_block(b), zero_format);

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
    // BCL::CSRMatrix<float, MKL_INT> a_local(dir + fname, BCL::FileFormat::Binary);
    // BCL::CSRMatrix<float, MKL_INT> b_local(dir + fname, BCL::FileFormat::Binary);
    BCL::CSRMatrix<float, MKL_INT> a_local(fname, mat_format);
    BCL::CSRMatrix<float, MKL_INT> b_local(fname, mat_format);

    printf("Multiplying locally...\n");
    fflush(stdout);
    auto begin = std::chrono::high_resolution_clock::now();
    auto c_local = a_local.dot_mkl(b_local);
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

template <typename T, typename index_type>
struct RemoteCSR {
  size_t m_;
  size_t n_;
  size_t nnz_;
  size_t i_;
  size_t j_;
  BCL::GlobalPtr<T> values_;
  BCL::GlobalPtr<index_type> row_ptr_;
  BCL::GlobalPtr<index_type> col_ind_;

  RemoteCSR() = default;
  RemoteCSR(const RemoteCSR&) = default;
  RemoteCSR& operator=(const RemoteCSR&) = default;
  RemoteCSR(RemoteCSR&&) = default;
  RemoteCSR& operator=(RemoteCSR&&) = default;

  RemoteCSR(size_t m, size_t n, size_t nnz, const std::vector<T>& values,
            const std::vector<index_type>& row_ptr,
            const std::vector<index_type>& col_ind,
            size_t i, size_t j)
            : m_(m), n_(n), nnz_(nnz), i_(i), j_(j) {
    values_ = BCL::alloc<T>(nnz);
    col_ind_ = BCL::alloc<index_type>(nnz);
    row_ptr_ = BCL::alloc<index_type>(m+1);

    std::copy(values.begin(), values.end(), values_.local());
    std::copy(col_ind.begin(), col_ind.end(), col_ind_.local());
    std::copy(row_ptr.begin(), row_ptr.end(), row_ptr_.local());
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  BCL::CSRMatrix<T, index_type, Allocator> get_matrix() const {
    BCL::CSRMatrix<T, index_type, Allocator> mat(m_, n_, nnz_);

    auto r1 = BCL::arget(values_, mat.vals_.data(), mat.vals_.size());
    auto r2 = BCL::arget(row_ptr_, mat.row_ptr_.data(), mat.row_ptr_.size());
    auto r3 = BCL::arget(col_ind_, mat.col_ind_.data(), mat.col_ind_.size());

    r1.wait();
    r2.wait();
    r3.wait();

    return std::move(mat);
  }

  template <typename Allocator = BCL::bcl_allocator<T>>
  BCL::future<BCL::CSRMatrix<T, index_type, Allocator>> arget_matrix() const {
    using allocator_traits = std::allocator_traits<Allocator>;
    using IAllocator = typename allocator_traits:: template rebind_alloc<index_type>;

    auto vals = BCL::arget<T, Allocator>(values_, nnz_);
    auto row_ptr = BCL::arget<index_type, IAllocator>(row_ptr_, m_+1);
    auto col_ind = BCL::arget<index_type, IAllocator>(col_ind_, nnz_);

    return BCL::future<BCL::CSRMatrix<T, index_type, Allocator>>(m_, n_, nnz_,
                                                                 std::move(vals),
                                                                 std::move(row_ptr),
                                                                 std::move(col_ind));
  }
};

template <typename T, typename index_type>
void async_gemm(const BCL::SPMatrix<T, index_type>& a,
                 const BCL::SPMatrix<T, index_type>& b,
                       BCL::SPMatrix<T, index_type>& c) {
  BCL::print("Doing GEMM -- A %d x %d, B %d x %d, C %d x %d\n",
             a.grid_shape()[0], a.grid_shape()[1],
             b.grid_shape()[0], b.grid_shape()[1],
             c.grid_shape()[0], c.grid_shape()[1]);
  // accumulated C's: a map of grid coordinates to sparse
  //                  matrices (for now, also maps)

  size_t k_splits = 1;

  std::vector<BCL::GlobalPtr<int>> claimants(c.grid_shape()[0]*c.grid_shape()[1]);

  BCL::print("Creating many to many distributor (%lu, %lu)...\n",
             c.grid_shape()[0]*c.grid_shape()[1]*a.grid_shape()[1], 4);
  using remote_csr_type = RemoteCSR<T, index_type>;
  BCL::ManyToManyDistributor<remote_csr_type, BCL::identity_serialize<remote_csr_type>> dist(
                      c.grid_shape()[0]*c.grid_shape()[1]*a.grid_shape()[1], 4);

  BCL::print("Initializing claimants pointers.\n");
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {
        claimants[i*c.grid_shape()[1] + j] = BCL::alloc<int>(1);
        *claimants[i*c.grid_shape()[1] + j] = 0;
      }
    }
  }

  BCL::print("Broadcasting.\n");
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      claimants[i*c.grid_shape()[1] + j]
        = BCL::broadcast(claimants[i*c.grid_shape()[1] + j], c.tile_locale(i, j));
      }
  }

  std::unordered_map<
                     std::pair<size_t, size_t>,
                     // BCL::SparseSPAAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     BCL::SparseHashAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     // BCL::SparseHeapAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     // BCL::CombBLASAccumulator<T, index_type>,
                     // BCL::EagerMKLAccumulator<T, index_type>,
                     BCL::djb2_hash<std::pair<size_t, size_t>>
                     >
                     accumulated_cs;

  BCL::print("Doing local compute.\n");
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {

        int result;
        do {
          result = BCL::fetch_and_op<int>(claimants[i*c.grid_shape()[1] + j], k_splits, BCL::plus<int>{});
          // If result is in range => you've claimed a segment of k.
          if (result < a.grid_shape()[1]) {
            // TODO: what should k_offset be?
            size_t k_offset = j % a.grid_shape()[1];

            size_t k_begin = result;
            size_t k_end = std::min(result + k_splits, a.grid_shape()[1]);

            auto begin = std::chrono::high_resolution_clock::now();
            auto buf_a = a.arget_tile(i, (k_begin+k_offset) % a.grid_shape()[1]);
            auto buf_b = b.arget_tile((k_begin+k_offset) % a.grid_shape()[1], j);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end - begin).count();
            issue += duration;

            for (size_t k_ = k_begin; k_ < k_end; k_++) {
              size_t k = (k_ + k_offset) % a.grid_shape()[1];

              begin = std::chrono::high_resolution_clock::now();
              auto my_a = buf_a.get();
              auto my_b = buf_b.get();
              end = std::chrono::high_resolution_clock::now();
              duration = std::chrono::duration<double>(end - begin).count();
              synct += duration;

              if (k_+1 < k_end) {
                begin = std::chrono::high_resolution_clock::now();
                buf_a = a.arget_tile(i, (k+1) % a.grid_shape()[1]);
                buf_b = b.arget_tile((k+1) % a.grid_shape()[1], j);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration<double>(end - begin).count();
                issue += duration;
              }

              begin = std::chrono::high_resolution_clock::now();
              auto c = my_a.dot(my_b);
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
        } while (result < a.grid_shape()[1]);
      }
    }
  }

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {

      if (c.tile_locale(i, j) != BCL::rank()) {
      int result;
      do {
        result = BCL::fetch_and_op<int>(claimants[i*c.grid_shape()[1] + j], k_splits, BCL::plus<int>{});
        // If result is in range => you've claimed a segment of k.
        if (result < a.grid_shape()[1]) {
          size_t k_offset = j % a.grid_shape()[1];

          size_t k_begin = result;
          size_t k_end = std::min(result + k_splits, a.grid_shape()[1]);

          fprintf(stderr, "%lu stole tile (%lu, %lu) k: %lu -> %lu from %lu\n",
                  BCL::rank(), i, j, k_begin, k_end, c.tile_locale(i, j));

          auto begin = std::chrono::high_resolution_clock::now();
          auto buf_a = a.arget_tile(i, (k_begin+k_offset) % a.grid_shape()[1]);
          auto buf_b = b.arget_tile((k_begin+k_offset) % a.grid_shape()[1], j);
          auto end = std::chrono::high_resolution_clock::now();
          double duration = std::chrono::duration<double>(end - begin).count();
          issue += duration;

          for (size_t k_ = k_begin; k_ < k_end; k_++) {
            size_t k = (k_ + k_offset) % a.grid_shape()[1];

            begin = std::chrono::high_resolution_clock::now();
            auto my_a = buf_a.get();
            auto my_b = buf_b.get();
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end - begin).count();
            synct += duration;

            if (k_+1 < k_end) {
              begin = std::chrono::high_resolution_clock::now();
              buf_a = a.arget_tile(i, (k+1) % a.grid_shape()[1]);
              buf_b = b.arget_tile((k+1) % a.grid_shape()[1], j);
              end = std::chrono::high_resolution_clock::now();
              duration = std::chrono::duration<double>(end - begin).count();
              issue += duration;
            }

            begin = std::chrono::high_resolution_clock::now();
            auto c = my_a.dot(my_b);
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
      } while (result < a.grid_shape()[1]);

      if (accumulated_cs.find({i, j}) != accumulated_cs.end()) {
        auto begin = std::chrono::high_resolution_clock::now();
        auto mat = accumulated_cs[{i, j}].get_matrix(c.tile_shape(i, j)[0],
                                                     c.tile_shape(i, j)[1]);
        auto r_mat = RemoteCSR<T, index_type>(mat.m_, mat.n_, mat.nnz_,
                                              mat.vals_, mat.row_ptr_, mat.col_ind_,
                                              i, j);
        dist.insert(r_mat, c.tile_locale(i, j));
        accumulated_cs.erase({i, j});
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - begin).count();
        acc += duration;
      }
      } // end if tile_locale(i, j) != BCL::rank()
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
  // BCL::barrier();
  dist.flush();
  BCL::print("Done with compute, accumulating...\n");
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  load += duration;

  begin = std::chrono::high_resolution_clock::now();

  BCL::print("Grabbing stolen accumulations...\n");
  std::vector<std::pair<std::pair<size_t, size_t>,
                        decltype(dist.begin().local()->arget_matrix())
                       >
             > futures;
  for (auto it = dist.begin().local(); it != dist.end().local(); it++) {
    auto& r_mat = *it;
    size_t i = r_mat.i_;
    size_t j = r_mat.j_;
    futures.emplace_back(std::make_pair(std::make_pair(i, j), r_mat.arget_matrix()));
  }

  for (auto& fut : futures) {
    auto mat = fut.second.get();
    size_t i = fut.first.first;
    size_t j = fut.first.second;

    accumulated_cs[{i, j}].accumulate(std::move(mat));
  }

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (accumulated_cs.find({i, j}) != accumulated_cs.end()) {
        /*
        auto c_local = c.get_tile(i, j);
        accumulated_cs[{i, j}].accumulate(c_local);
        */

        auto cmatrix = accumulated_cs[{i, j}].get_matrix(c.tile_shape(i, j)[0], c.tile_shape(i, j)[1]);

        c.assign_tile(i, j, cmatrix, true);
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
