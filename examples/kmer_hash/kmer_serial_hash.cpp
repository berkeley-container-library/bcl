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

bool is_even(size_t val) {
  return (val % 2) == 0;
}

// XXX: fairly break tie between i and j.  Returns true if i wins.

bool break_tie(size_t i, size_t j, size_t N) {
  size_t min = std::min(i, j);
  size_t max = std::max(i, j);

  if ((is_even(max) && max / 2 > min) || (!is_even(max) && max / 2 <= min)) {
    if (min == i) {
      return true;
    } else {
      return false;
    }
  } else {
    if (max == i) {
      return true;
    } else {
      return false;
    }
  }
}

template <typename HashMap>
std::list<std::list<kmer_pair>> walk_right(HashMap& kmer_hash,
                                           const std::vector<kmer_pair>& start_nodes) {
  std::list<std::list<kmer_pair>> contigs;
  for (const auto& start_kmer : start_nodes) {

    assert(kmer_hash.find(start_kmer.kmer) != kmer_hash.end());

    if (kmer_hash[start_kmer.kmer].used != -1) {
      continue;
    }
    kmer_hash[start_kmer.kmer].used = 1;

    std::list<kmer_pair> contig = {start_kmer};

    std::unordered_set<pkmer_t> used_kmers;
    used_kmers.insert(pkmer_t(canonicalize(start_kmer.kmer_str())));
    bool success = true;
    bool aborted = false;

    while (success && contig.back().forwardExt() != 'F' &&
                      contig.back().forwardExt() != 'X') {
      pkmer_t next_kmer = pkmer_t(canonicalize(contig.back().next_kmer().get()));
      if (used_kmers.find(next_kmer) != used_kmers.end()) {
        break;
      }

      auto value = kmer_hash.find(next_kmer);
      assert(value != kmer_hash.end());

      if (value->second.used != -1) {
        aborted = true;
      }

      fb_ext fb = value->second;

      if (next_kmer.get() != contig.back().next_kmer().get()) {
        std::string new_exts = rcomplement(fb.get());
        fb_ext new_fb(new_exts);
        fb.init(new_fb.ext);
      }

      if (contig.back().kmer_str()[0] == fb.backwardExt()) {
        used_kmers.insert(next_kmer);
        contig.push_back(kmer_pair(contig.back().next_kmer().get(),
                                   fb.get()));
        value->second.used = 1;
      } else {
        aborted = false;
        success = false;
      }
    }

    if (!aborted) {
      contigs.push_back(contig);
    }
  }
  return contigs;
}

template <typename HashMap>
std::list<std::list<kmer_pair>> walk_left(HashMap& kmer_hash,
                                          const std::vector<kmer_pair>& start_nodes) {
  std::list<std::list<kmer_pair>> contigs;
  for (const auto& start_kmer : start_nodes) {

    assert(kmer_hash.find(start_kmer.kmer) != kmer_hash.end());

    // TODO: check if start_kmer is used?
    if (kmer_hash[start_kmer.kmer].used != -1) {
      continue;
    }
    kmer_hash[start_kmer.kmer].used = 1;

    std::list<kmer_pair> contig = {start_kmer};

    std::unordered_set<pkmer_t> used_kmers;
    used_kmers.insert(pkmer_t(canonicalize(start_kmer.kmer_str())));
    bool success = true;
    bool aborted = false;

    while (success && contig.front().backwardExt() != 'F' &&
                      contig.front().backwardExt() != 'X') {
      pkmer_t last_kmer = pkmer_t(canonicalize(contig.front().last_kmer().get()));
      if (used_kmers.find(last_kmer) != used_kmers.end()) {
        break;
      }

      auto value = kmer_hash.find(last_kmer);
      assert(value != kmer_hash.end());

      if (value->second.used != -1) {
        aborted = true;
      }

      fb_ext fb = value->second;

      if (last_kmer.get() != contig.front().last_kmer().get()) {
        std::string new_exts = rcomplement(fb.get());
        fb_ext new_fb(new_exts);
        fb.init(new_fb.ext);
      }

      if (contig.front().kmer_str().back() == fb.forwardExt()) {
        used_kmers.insert(last_kmer);
        contig.push_front(kmer_pair(contig.front().last_kmer().get(),
                                    fb.get()));
        value->second.used = 1;
      } else {
        aborted = false;
        success = false;
      }
    }
    if (!aborted) {
      contigs.push_back(contig);
    }
  }
  return contigs;
}

int main (int argc, char **argv) {
  BCL::init(16384);

  std::string kmer_fname = "/home/ubuntu/data/kmer/human-new.dat";

  assert(BCL::nprocs() == 1);

  size_t n_kmers = line_count(kmer_fname);

  printf("Got %lu k-mers\n", n_kmers);

  std::unordered_map<pkmer_t, fb_ext> kmer_hash;

  double load_factor = 0.5;
  size_t hash_table_size = n_kmers * (1.0 / load_factor);

  kmer_hash.reserve(n_kmers);

  printf("Initializing hash table of size %lu for %lu kmers.\n",
         hash_table_size, n_kmers);

  std::vector <kmer_pair> kmers = read_kmers(kmer_fname, BCL::nprocs(), BCL::rank());

  size_t canonicalized = 0;
  for (size_t i = 0; i < kmers.size(); i++) {
    if (kmers[i].kmer_str() != canonicalize(kmers[i].kmer_str())) {
      std::string kmer_str = canonicalize(kmers[i].kmer_str());
      std::string fb = rcomplement(kmers[i].fb_ext_str());
      kmers[i] = kmer_pair(kmer_str, fb);
      canonicalized++;
    }
  }

  printf("%lu k-mers canonicalized\n", canonicalized);

  printf("Finished reading k-mers.\n");

  std::vector <kmer_pair> right_start_nodes;
  std::vector <kmer_pair> left_start_nodes;

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto& kmer : kmers) {
    fb_ext fb(kmer.fb_ext);
    assert(kmer_hash.find(kmer.kmer) == kmer_hash.end());
    kmer_hash.insert({kmer.kmer, fb});
    if (kmer.backwardExt() == 'F' || kmer.backwardExt() == 'X') {
      right_start_nodes.push_back(kmer);
    }
    if (kmer.forwardExt() == 'F' || kmer.forwardExt() == 'X') {
      left_start_nodes.push_back(kmer);
    }
  }
  auto end_insert = std::chrono::high_resolution_clock::now();

  double insert_time = std::chrono::duration<double>(end_insert - start).count();
  printf("Finished inserting in %lf\n", insert_time);

  using CT = std::list<std::list<kmer_pair>>;

  CT contigs;

  auto start_read = std::chrono::high_resolution_clock::now();

  CT right_tigs = walk_right(kmer_hash, right_start_nodes);
  CT left_tigs = walk_left(kmer_hash, left_start_nodes);

  CT right_wild_tigs;

  BCL::barrier();

  for (const auto& kp : kmer_hash) {
    pkmer_t kmer = kp.first;
    fb_ext fb = kp.second;

    kmer_pair my_kmer(kmer.get(), fb.get());

    if (my_kmer.backwardExt() != 'F' && my_kmer.backwardExt() != 'X') {
      pkmer_t last_kmer(canonicalize(my_kmer.last_kmer().get()));

      auto value = kmer_hash.find(last_kmer);

      if (value == kmer_hash.end()) {
        CT wild_tig = walk_right(kmer_hash, {my_kmer});
        right_wild_tigs.splice(right_wild_tigs.end(), wild_tig);
      } else {
        fb_ext fb = value->second;

        if (last_kmer.get() != my_kmer.last_kmer().get()) {
          std::string new_exts = rcomplement(fb.get());
          fb_ext new_fb(new_exts);
          fb.init(new_fb.ext);
        }
        if (my_kmer.kmer_str().back() != fb.forwardExt()) {
          CT wild_tig = walk_right(kmer_hash, {my_kmer});
          right_wild_tigs.splice(right_wild_tigs.end(), wild_tig);
        }
      }
    }
  }

  CT left_wild_tigs;

  for (const auto& kp : kmer_hash) {
    pkmer_t kmer = kp.first;
    fb_ext fb = kp.second;

    kmer_pair my_kmer(kmer.get(), fb.get());

    if (my_kmer.forwardExt() != 'F' && my_kmer.forwardExt() != 'X') {
      pkmer_t next_kmer(canonicalize(my_kmer.next_kmer().get()));

      auto value = kmer_hash.find(next_kmer);
      
      if (value == kmer_hash.end()) {
        CT wild_tig = walk_left(kmer_hash, {my_kmer});
        left_wild_tigs.splice(left_wild_tigs.end(), wild_tig);
      } else {
        fb_ext fb = value->second;

        if (next_kmer.get() != my_kmer.next_kmer().get()) {
          std::string new_exts = rcomplement(fb.get());
          fb_ext new_fb(new_exts);
          fb.init(new_fb.ext);
        }
        if (my_kmer.kmer_str()[0] != fb.backwardExt()) {
          CT wild_tig = walk_left(kmer_hash, {my_kmer});
          left_wild_tigs.splice(left_wild_tigs.end(), wild_tig);
        }
      }
    }
  }

  auto end_read = std::chrono::high_resolution_clock::now();
  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  size_t n_right_tigs = BCL::allreduce<size_t>(right_tigs.size(), std::plus<size_t>{});
  size_t n_left_tigs = BCL::allreduce<size_t>(left_tigs.size(), std::plus<size_t>{});
  size_t n_wright_tigs = BCL::allreduce<size_t>(right_wild_tigs.size(), std::plus<size_t>{});
  size_t n_wleft_tigs = BCL::allreduce<size_t>(left_wild_tigs.size(), std::plus<size_t>{});

  contigs.splice(contigs.end(), right_tigs);
  contigs.splice(contigs.end(), left_tigs);
  contigs.splice(contigs.end(), right_wild_tigs);
  contigs.splice(contigs.end(), left_wild_tigs);

  BCL::print("%llu, %llu, %llu, %llu\n", n_right_tigs, n_left_tigs, n_wright_tigs,
    n_wleft_tigs);

  std::chrono::duration<double> read = end_read - start_read;
  std::chrono::duration<double> insert = end_insert - start;
  std::chrono::duration<double> total = end - start;

  size_t numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (size_t sum, const std::list <kmer_pair> &contig) {
      return sum + contig.size();
    });

  printf("Rank %lu reconstructed %lu contigs with %lu nodes from %lu start nodes. "
    "(%lf read, %lf insert, %lf total)\n", BCL::rank(), contigs.size(),
    numKmers, right_start_nodes.size() + left_start_nodes.size(), read.count(),
    insert.count(), total.count());

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
    std::string my_contig = extract_contig(contig);
    std::string canonicalized_contig = canonicalize(my_contig.substr(1, my_contig.length()-2));
    if (canonicalized_contig.length() >= KMER_LEN) {
      fout << canonicalized_contig << std::endl;
    }
  }
  fout.close();

  BCL::finalize();
  return 0;
}
