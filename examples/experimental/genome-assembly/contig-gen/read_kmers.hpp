#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>

#include "kmer_t.hpp"

size_t line_count(const std::string &fname) {
  FILE *f = fopen(fname.c_str(), "r");

  if (f == NULL) {
    throw std::runtime_error("line_count: could not open " + fname);
  }

  size_t n_lines = 0;
  size_t n_read;

  const size_t buf_size = 16384;
  char buf[buf_size];

  do {
    n_read = fread(buf, sizeof(char), buf_size, f);
    for (size_t i = 0; i < n_read; i++) {
      if (buf[i] == '\n') {
        n_lines++;
      }
    }
  } while (n_read != 0);
  fclose(f);
  return n_lines;
}

std::vector <kmer_pair> read_kmers(const std::string &fname, uint64_t nprocs = 1, uint64_t rank = 0) {
  size_t num_lines = line_count(fname);
  size_t split = (num_lines + nprocs - 1) / nprocs;
  size_t start = split*rank;
  size_t size = std::min(split, num_lines - start);

  FILE *f = fopen(fname.c_str(), "r");
  const size_t line_len = KMER_LEN + 4;
  fseek(f, line_len*start, SEEK_SET);

  std::shared_ptr <char> buf(new char[line_len*size]);
  size_t n_read = fread(buf.get(), sizeof(char), line_len*size, f);

  std::vector <kmer_pair> kmers;

  for (size_t line_offset = 0; line_offset < line_len*size; line_offset += line_len) {
    char *kmer_buf = &buf.get()[line_offset];
    char *fb_ext_buf = kmer_buf + KMER_LEN+1;
    kmers.push_back(kmer_pair(std::string(kmer_buf, KMER_LEN), std::string(fb_ext_buf, 2)));
  }
  fclose(f);
  return kmers;
}

// Extract contig for the full application version of contig generation.
std::string extract_contig_fullapp(const std::list <kmer_pair> &contig) {
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

// Extract contig for the CS 267 version of contig generation.
std::string extract_contig_simple(const std::list <kmer_pair> &contig) {
  std::string contig_buf = "";

  contig_buf += contig.front().kmer_str();

  for (const auto &kmer : contig) {
    if (kmer.forwardExt() != 'F') {
      contig_buf += kmer.forwardExt();
    }
  }
  return contig_buf;
}
