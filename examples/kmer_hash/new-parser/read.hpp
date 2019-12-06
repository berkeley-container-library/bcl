#pragma once

#include "readufx.cpp"
#include "kmer_t.hpp"

#include <bcl/bcl.hpp>

std::vector <kmer_pair_> read_kmers(const std::string &fname) {
  FILE *fdInput;
  int dsize;
  int64_t size, myshare;
  char **kmersarr;
  char *lefts;
  char *rights;
  int *counts;

  int dmin = 4;

  std::vector <kmer_pair_> kmers;

  fdInput = UFXInitOpen(fname.c_str(), &dsize, &myshare, BCL::nprocs(), BCL::rank(), &size);

  int64_t my_ufx_lines;
  if (BCL::rank() == BCL::nprocs()-1) {
    my_ufx_lines = (size / BCL::nprocs() + size % BCL::nprocs());
  } else {
    my_ufx_lines = size / BCL::nprocs();
  }

  int64_t kmers_read = UFXRead(fdInput, dsize, &kmersarr, &counts, &lefts, &rights, my_ufx_lines, dmin, 0, BCL::rank(), kmers);

  DeAllocateAll(&kmersarr, &counts, &lefts, &rights, kmers_read);

  fclose(fdInput);
  return kmers;
}


std::string extract_contig(const std::list <kmer_pair> &contig) {
  std::string contig_buf = "";

  if (contig.front().backwardExt() != 'X' && contig.front().backwardExt() != 'F') {
    contig_buf += contig.front().backwardExt();
  }
  contig_buf += contig.front().kmer_str();

  for (const auto &kmer : contig) {
    if (kmer.forwardExt() != 'F' && kmer.forwardExt() != 'X') {
      contig_buf += kmer.forwardExt();
    }
  }
  return canonicalize(contig_buf);
}
