#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <limits>
#include <numeric>

#include <unordered_set>

#include <bcl/bcl.hpp>

#include "kmer_t.hpp"
#include "read.hpp"

#include <bcl/containers/HashMap.hpp>
#include <bcl/containers/HashMapBuffer.hpp>

#include <futar/while.hpp>
#include <futar/pools/ChainPool.hpp>

  template <typename T>
  struct wrapper {
    wrapper(T&& val) : val_(std::move(val)) {}

    template <std::size_t I = 0>
    void get_impl_() {
      using current_type = decltype(std::get<I>(val_));
      if (val_.index() == I) {
        if constexpr(futar::is_future<current_type>::value) {
          std::get<I>(val_).get();
        }
      }

      if constexpr(I+1 < std::variant_size<decltype(val_)>::value) {
        get_impl_<I+1>();
      }
    }

    void get() {
      get_impl_();
    }
    std::variant<T, bool> val_;
  };

struct fb_ext {
  char ext[2];
  int used = 0;

  fb_ext(const char ext[2]) {
    init(ext);
  }

  fb_ext(const fb_ext &ext) {
    this->used = ext.used;
    init(ext.ext);
  }

  fb_ext(const std::string ext) {
    init(ext.data());
  }

  fb_ext operator=(const fb_ext &ext) {
    this->used = ext.used;
    init(ext.ext);
    return *this;
  }

  fb_ext() {}

  std::string get() const noexcept {
    return std::string(ext, 2);
  }

  char backwardExt() const noexcept {
    return ext[0];
  }

  char forwardExt() const noexcept {
    return ext[1];
  }

  void init(const char ext[2]) {
    for (int i = 0; i < 2; i++)
      this->ext[i] = ext[i];
  }
};

namespace std {
  template <>
  struct hash <pkmer_t> {
    size_t operator()(const pkmer_t &pkmer) const noexcept {
      return pkmer.hash();
    }
  };
}

int main (int argc, char **argv) {
  BCL::init(1024);

  std::string kmer_fname = "/global/cscratch1/sd/brock/large-kmerdata/human-chr14.txt.ufx.bin";
  // std::string kmer_fname = "../small.dat";

  if (argc < 2) {
    BCL::print("usage: ./simplified_kmer [concurrency]\n");
    BCL::finalize();
    return 1;
  }

  size_t concurrency = std::atoll(argv[1]);

  BCL::print("Reading k-mers...\n");
  std::vector <kmer_pair_> kmers = read_kmers(kmer_fname);

  size_t n_kmers = kmers.size();

  n_kmers = BCL::allreduce <uint64_t> (n_kmers, std::plus <uint64_t> ());
  BCL::print("Got %llu k-mers\n", n_kmers);

  // Load factor of 0.7
  size_t hash_table_size = n_kmers * (1.0 / 0.5);


  BCL::HashMap <Kmer, fb_ext,
    BCL::identity_serialize <Kmer>,
    BCL::identity_serialize <fb_ext>> kmer_hash(hash_table_size);

  BCL::print("Initializing hash table of size %d for %d kmers.\n", hash_table_size,
    n_kmers);


  BCL::print("Finished reading k-mers.\n");

  BCL::barrier();

  std::vector<kmer_pair_> right_kmers;
  std::vector<kmer_pair_> left_kmers;

  BCL::HashMapBuffer <Kmer, fb_ext,
    BCL::identity_serialize <Kmer>,
    BCL::identity_serialize <fb_ext>> kmer_hash_buffer(kmer_hash, 2*(n_kmers / BCL::nprocs()), 200);

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto &kmer : kmers) {
    fb_ext fb(kmer.fb_ext());
    // bool success = kmer_hash.insert(kmer.kmer(), fb);
    bool success = kmer_hash_buffer.insert(kmer.kmer(), fb);
    if (!success) {
      throw std::runtime_error("BCL HashMapBuffer: not enough space");
    }
    if (kmer.backwardExt() == 'F' || kmer.backwardExt() == 'X') {
      right_kmers.push_back(kmer);
    } else if (kmer.forwardExt() == 'F' || kmer.forwardExt() == 'X') {
      left_kmers.push_back(kmer);
    }
  }
  kmer_hash_buffer.flush();
  // BCL::barrier();
  auto end_insert = std::chrono::high_resolution_clock::now();

  double insert_time = std::chrono::duration <double> (end_insert - start).count();
  BCL::print("Finished inserting in %lf\n", insert_time);

  using CT = typename std::list <std::list <kmer_pair_>>;

  CT contigs;

  futar::FuturePool<bool> pool(concurrency);

  BCL::print("About to traverse...\n");

  auto start_read = std::chrono::high_resolution_clock::now();

  /*
  for (const auto& kmer : right_kmers) {
    contigs.emplace_back();
    std::list<kmer_pair_>& contig = contigs.back();
    contig.push_back(kmer);

    std::unordered_set<Kmer> kmer_map;
    kmer_map.insert(contig.back().kmer().rep());

    while (contig.back().forwardExt() != 'F' && contig.back().forwardExt() != 'X') {
      Kmer next_kmer = contig.back().next_kmer().rep();
      if (kmer_map.find(next_kmer) != kmer_map.end()) {
        break;
      } else {
        kmer_map.insert(next_kmer);
      }
      auto fb = kmer_hash.arfind(next_kmer).get();

      if (!fb.has_value()) {
        break;
      } else {
        std::string fb_;
        if (next_kmer != contig.back().next_kmer()) {
          fb_ = rcomplement(fb.value().get());
        } else {
          fb_ = fb.value().get();
        }
        kmer_pair_ kmer(contig.back().next_kmer(), fb_);
        contig.emplace_back(kmer);
      }
    }
  }
  */

  std::list<std::unordered_set<Kmer>> used_sets;
  for (const auto& kmer : right_kmers) {
    contigs.emplace_back();
    used_sets.emplace_back();

    std::list<kmer_pair_>& contig = contigs.back();
    std::unordered_set<Kmer>& kmer_map = used_sets.back();

    contig.push_back(kmer);
    kmer_map.insert(contig.back().kmer().rep());

    futar::while_ fut([&]() -> bool {
                        Kmer next_kmer = contig.back().next_kmer().rep();
                        bool not_found = (kmer_map.find(next_kmer) == kmer_map.end());
                        bool continue_ = not_found && contig.back().forwardExt() != 'F' && contig.back().forwardExt() != 'X';
                        if (!continue_) {
                          kmer_map.clear();
                        }
                        return continue_;
                      },
                      [&]() {
                        return futar::call([&](auto fb) {
                          if (!fb.has_value()) {
                            kmer_pair_ kmer(contig.back().next_kmer(), "FF");
                            contig.emplace_back(kmer);
                          } else {
                            std::string fb_;
                            if (contig.back().next_kmer().rep() != contig.back().next_kmer()) {
                              fb_ = rcomplement(fb.value().get());
                            } else {
                              fb_ = fb.value().get();
                            }
                            kmer_pair_ kmer(contig.back().next_kmer(), fb_);
                            kmer_map.insert(contig.back().next_kmer().rep());
                            contig.emplace_back(kmer);
                          }
                          return true;
                        }, kmer_hash.arfind(contig.back().next_kmer().rep()));
                      });

    pool.push_back(std::move(fut));
  }

  pool.drain();

  auto end_read = std::chrono::high_resolution_clock::now();
  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration <double> read = end_read - start_read;
  std::chrono::duration <double> insert = end_insert - start;
  std::chrono::duration <double> total = end - start;

  size_t numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (size_t sum, const auto& contig) {
      return sum + contig.size();
    });

  printf("Rank %lu reconstructed %lu contigs with %d nodes from %lu start nodes. "
    "(%lf read, %lf insert, %lf total)\n", BCL::rank(), contigs.size(),
    numKmers, right_kmers.size(),
    read.count(), insert.count(), total.count());

  size_t totalKmers = BCL::allreduce<size_t>(numKmers, std::plus<size_t>{});

  size_t totalStartKmers = BCL::allreduce<size_t>(right_kmers.size(), std::plus<size_t>{});

  BCL::print("Assembled in %lf total. (%lf, %lf)\n", total.count(), insert.count(),
    read.count());

  BCL::print("Assembled %lu k-mers from %lu start k-mers total.\n", totalKmers,
             totalStartKmers);

  BCL::finalize();
  return 0;
}
