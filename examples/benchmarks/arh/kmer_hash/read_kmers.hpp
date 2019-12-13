#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

#include "kmer_t.hpp"

// Return the size of the k-mers in fname
int kmer_size(const std::string &fname) {
  std::ifstream fin(fname);
  if (!fin.is_open()) {
    throw std::runtime_error("kmer_size: could not open " + fname);
  }

  std::string buf;
  fin >> buf;
  fin.close();

  return buf.size();
}


// Get the number of lines in fname
size_t line_count(const std::string &fname) {
  // speedup the process
  static const std::unordered_map<std::string, size_t> line_num_cache({
      {"/global/project/projectdirs/mp309/cs267-spr2018/hw3-datasets/test.txt", -1},
      {"/global/cscratch1/sd/jackyan/my_datasets/test.txt", -1},
      {"/global/project/projectdirs/mp309/cs267-spr2018/hw3-datasets/large.txt", 27000544},
      {"/global/cscratch1/sd/jackyan/my_datasets/large.txt", 27000544},
      {"/global/project/projectdirs/mp309/cs267-spr2018/hw3-datasets/human-chr14-synthetic.txt", 89710742},
      {"/global/cscratch1/sd/jackyan/my_datasets/human-chr14-synthetic.txt", 89710742},
      {"/global/project/projectdirs/mp309/cs267-spr2018/synthetic-hw3/kmers_500_200.dat", 102400000},
      {"/global/cscratch1/sd/jackyan/my_datasets/kmers_500_200.dat", 102400000}
  });
  auto got = line_num_cache.find(fname);
  if (got != line_num_cache.end() && got->second != -1) {
    return got->second;
  }

  FILE *f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    throw std::runtime_error("line_count: could not open " + fname);
  }
  size_t n_lines = 0;
  int n_read;

  const size_t buf_size = 16384;
  char buf[buf_size];

  do {
    n_read = fread(buf, sizeof(char), buf_size, f);
    for (int i = 0; i < n_read; i++) {
      if (buf[i] == '\n') {
        n_lines++;
      }
    }
  } while (n_read != 0);
  fclose(f);
  return n_lines;
}

// Read k-mers from fname.
// If nprocs and rank are given, each rank will read
// an appropriately sized block portion of the k-mers.
std::vector <kmer_pair> read_kmers(const std::string &fname, size_t num_lines, int nprocs = 1, int rank = 0) {
  size_t split = (num_lines + nprocs - 1) / nprocs;
  size_t start = split*rank;
  size_t size = std::min(split, num_lines - start);

  FILE *f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    throw std::runtime_error("read_kmers: could not open " + fname);
  }
  const size_t line_len = KMER_LEN + 4;
  fseek(f, line_len*start, SEEK_SET);

  std::shared_ptr <char> buf(new char[line_len*size]);
  fread(buf.get(), sizeof(char), line_len*size, f);

  std::vector <kmer_pair> kmers;

  for (size_t line_offset = 0; line_offset < line_len*size; line_offset += line_len) {
    char *kmer_buf = &buf.get()[line_offset];
    char *fb_ext_buf = kmer_buf + KMER_LEN+1;
    kmers.push_back(kmer_pair(std::string(kmer_buf, KMER_LEN), std::string(fb_ext_buf, 2)));
  }
  fclose(f);
  return kmers;
}

std::string extract_contig(const std::list <kmer_pair> &contig) {
  std::string contig_buf = "";

  contig_buf += contig.front().kmer_str();

  for (const auto &kmer : contig) {
    if (kmer.forwardExt() != 'F') {
      contig_buf += kmer.forwardExt();
    }
  }
  return contig_buf;
}
