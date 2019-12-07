#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <set>
#include <numeric>
#include <cstddef>
#include <chrono>
#include <upcxx/upcxx.hpp>

#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "hash_map_stl.hpp"

#include "butil.hpp"

int main(int argc, char **argv) {
  upcxx::init();

  if (argc < 2) {
    BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test|verbose_test]\n");
    upcxx::finalize();
    exit(1);
  }

  std::string kmer_fname = std::string(argv[1]);
  std::string run_type = "";

  if (argc >= 3) {
    run_type = std::string(argv[2]);
  }

  int ks = kmer_size(kmer_fname);

  if (ks != KMER_LEN) {
    throw std::runtime_error("Error: " + kmer_fname + " contains " +
      std::to_string(ks) + "-mers, while this binary is compiled for " +
      std::to_string(KMER_LEN) + "-mers.  Modify packing.hpp and recompile.");
  }
  
  size_t n_kmers = line_count(kmer_fname);
  HashMap hashmap(n_kmers);

  std::vector <kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

  if (run_type == "verbose" || run_type == "verbose_test") {
    BUtil::print("Finished reading kmers.\n");
  }

  auto start = std::chrono::high_resolution_clock::now();
  
  std::vector <kmer_pair> start_nodes;
  
  upcxx::future<> fut_all = upcxx::make_future();
  for (auto &kmer : kmers) 
    fut_all = upcxx::when_all(fut_all, hashmap.insert(kmer));
  for (auto& kmer : kmers) if (kmer.backwardExt() == 'F') {
    start_nodes.push_back(kmer);
  }
  fut_all.wait();

  auto end_insert = std::chrono::high_resolution_clock::now();
  upcxx::barrier();

  double insert_time = std::chrono::duration <double> (end_insert - start).count();
  if (run_type != "test") {
    BUtil::print("Finished inserting in %lf\n", insert_time);
  }
  upcxx::barrier();

  auto start_read = std::chrono::high_resolution_clock::now();

  std::list <std::list <kmer_pair>> contigs;

  upcxx::barrier();
  for (const auto &start_kmer : start_nodes) {
    std::list <kmer_pair> contig;
    contig.push_back(start_kmer);
    while (contig.back().forwardExt() != 'F') {
      kmer_pair kmer;
      if ((kmer = hashmap.find(contig.back().next_kmer()).wait())
           == kmer_pair()) {
        throw std::runtime_error("Error: k-mer not found in hashmap.");
        printf("Error in rank %d\n", upcxx::rank_me());
      }
      contig.push_back(kmer);
    }
    contigs.push_back(contig);
  }
  upcxx::barrier();

  auto end_read = std::chrono::high_resolution_clock::now();
  upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration <double> read = end_read - start_read;
  std::chrono::duration <double> insert = end_insert - start;
  std::chrono::duration <double> total = end - start;

  int numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (int sum, const std::list <kmer_pair> &contig) {
      return sum + contig.size();
    });

  if (run_type != "test") {
    BUtil::print("Assembled in %lf total\n", total.count());
  }
  
  if (run_type == "verbose" || run_type == "verbose_test") {
    printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
      " (%lf read, %lf insert, %lf total)\n", upcxx::rank_me(), contigs.size(),
      numKmers, start_nodes.size(), read.count(), insert.count(), total.count());
  }

  if (run_type == "test" || run_type == "verbose_test") {
    std::ofstream fout("test_" + std::to_string(upcxx::rank_me()) + ".dat");
    for (const auto &contig : contigs) {
      fout << extract_contig(contig) << std::endl;
    }
    fout.close();
  }

  upcxx::barrier();
  upcxx::finalize();
  return 0;
}