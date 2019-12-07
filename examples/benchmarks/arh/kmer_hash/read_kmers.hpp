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
std::vector <kmer_pair> read_kmers(const std::string &fname, int nprocs = 1, int rank = 0) {
  size_t num_lines = line_count(fname);
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
