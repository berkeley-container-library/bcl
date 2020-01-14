#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <limits>
#include <numeric>

#include <unordered_set>

#include <bcl/bcl.hpp>

#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include <bcl/containers/HashMap.hpp>
#include <bcl/containers/HashMapBuffer.hpp>

struct fb_ext {
  char ext[2];
  int used = -1;

  fb_ext(const char ext[2]) {
    init(ext);
  }

  fb_ext(const std::string ext) {
    init(ext.data());
  }

  fb_ext() = default;
  fb_ext(const fb_ext&) = default;
  fb_ext& operator=(const fb_ext&) = default;

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
  BCL::init(2048);

  std::string kmer_fname = "/global/cscratch1/sd/brock/267-dataset/large.txt";

  size_t n_kmers = 0;
  if (BCL::rank() == 0) {
    n_kmers = line_count(kmer_fname);
  }

  n_kmers = BCL::allreduce<size_t>(n_kmers, std::plus<size_t>{});

  BCL::print("Got %lu k-mers\n", n_kmers);

  double load_factor = 0.5;
  size_t hash_table_size = n_kmers * (1.0 / load_factor);

  BCL::HashMap <pkmer_t, fb_ext,
                std::hash<pkmer_t>,
                BCL::identity_serialize <pkmer_t>,
                BCL::identity_serialize <fb_ext>
               > kmer_hash(hash_table_size);

  BCL::print("Initializing hash table of size %lu for %lu kmers.\n",
             hash_table_size, n_kmers);

  std::vector <kmer_pair> kmers = read_kmers(kmer_fname, BCL::nprocs(), BCL::rank());

  BCL::print("Finished reading k-mers.\n");

  BCL::barrier();

  std::vector<kmer_pair> start_nodes;

  BCL::HashMapBuffer <pkmer_t, fb_ext,
                      std::hash<pkmer_t>,
                      BCL::identity_serialize <pkmer_t>,
                      BCL::identity_serialize <fb_ext>
                     > kmer_hash_buffer(kmer_hash, 2*(n_kmers / BCL::nprocs()), 200);

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto& kmer : kmers) {
    fb_ext fb(kmer.fb_ext);
    // bool success = kmer_hash.insert_atomic_impl_(kmer.kmer, fb);
    bool success = kmer_hash_buffer.insert(kmer.kmer, fb);
    if (!success) {
      throw std::runtime_error("BCL HashMapBuffer: not enough space");
    }
    if (kmer.backwardExt() == 'F') {
      start_nodes.push_back(kmer);
    }
  }
  kmer_hash_buffer.flush();
  // BCL::barrier();
  auto end_insert = std::chrono::high_resolution_clock::now();

  double insert_time = std::chrono::duration<double>(end_insert - start).count();
  BCL::print("Finished inserting in %lf\n", insert_time);

  using CT = std::list<std::list<kmer_pair>>;

  CT contigs;

  auto start_read = std::chrono::high_resolution_clock::now();

  for (const auto& start_kmer : start_nodes) {
    std::list<kmer_pair> contig;
    contig.push_back(start_kmer);
    while (contig.back().forwardExt() != 'F') {
      pkmer_t next_kmer = pkmer_t(contig.back().next_kmer().get());
      auto kmer_iter = kmer_hash.find(next_kmer, BCL::HashMapAL::find);
      fb_ext fb = *kmer_iter;
      kmer_pair kmer(next_kmer.get(), fb.get());
      contig.push_back(kmer);
    }
    contigs.push_back(contig);
  }

  BCL::barrier();
  auto end_read = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> read = end_read - start_read;
  std::chrono::duration<double> insert = end_insert - start;
  std::chrono::duration<double> total = end - start;

  size_t numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (size_t sum, const std::list <kmer_pair> &contig) {
      return sum + contig.size();
    });

  printf("Rank %lu reconstructed %lu contigs with %lu nodes from %lu start nodes. "
    "(%lf read, %lf insert, %lf total)\n",
    BCL::rank(),
    contigs.size(), numKmers, start_nodes.size(),
    read.count(), insert.count(), total.count());

  size_t numBases = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (size_t sum, const std::list <kmer_pair> &contig) {
      return sum + KMER_LEN + contig.size() - 1;
    });

  size_t totalKmers = BCL::allreduce(numKmers, std::plus<size_t>{});
  size_t totalContigs = BCL::allreduce(contigs.size(), std::plus<size_t>{});
  size_t totalBases = BCL::allreduce(numBases, std::plus<size_t>{});
  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  BCL::print("Assembled in %lf total. (%lf, %lf)\n", total.count(), insert.count(),
    read.count());

  std::ofstream fout("test_" + std::to_string(BCL::rank()) + ".dat");
  for (const auto &contig : contigs) {
    std::string my_contig = extract_contig_simple(contig);
    if (my_contig.length() >= KMER_LEN) {
      fout << my_contig << std::endl;
    }
  }
  fout.close();

  BCL::finalize();
  return 0;
}
