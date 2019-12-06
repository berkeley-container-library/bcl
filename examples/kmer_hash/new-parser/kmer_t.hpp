#pragma once

#include "packing.hpp"
#include "hash_funcs.hpp"

#include "Kmer.hpp"

#include <algorithm>
#include <string>
#include <cstring>

std::string rcomplement(std::string str) {
  std::string rstr = std::string(str);
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == 'A') {
      rstr[i] = 'T';
    }
    if (str[i] == 'C') {
      rstr[i] = 'G';
    }
    if (str[i] == 'G') {
      rstr[i] = 'C';
    }
    if (str[i] == 'T') {
      rstr[i] = 'A';
    }
  }
  std::reverse(rstr.begin(), rstr.end());
  return rstr;
}

std::string canonicalize(std::string str) {
  std::string reverse = rcomplement(str);
  int lex_ind = strncmp(str.data(), reverse.data(), str.length());
  if (lex_ind < 0) {
    return str;
  } else {
    return reverse;
  }
}

struct kmer_pair_ {
  Kmer kmer_;
  char fb_ext_[2];

  kmer_pair_() = default;
  ~kmer_pair_() = default;

  kmer_pair_(const kmer_pair_&) = default;
  kmer_pair_(kmer_pair_&&) = default;

  kmer_pair_(const Kmer& kmer, const std::string& fb_ext) : kmer_(kmer) {
    assert(fb_ext.size() == 2);
    fb_ext_[0] = fb_ext[0];
    fb_ext_[1] = fb_ext[1];
  }

  kmer_pair_(const std::string& kmer, const std::string& fb_ext) : kmer_(kmer) {
    assert(fb_ext.size() == 2);
    fb_ext_[0] = fb_ext[0];
    fb_ext_[1] = fb_ext[1];
  }

  auto kmer() const noexcept {
    return kmer_;
  }

  char backwardExt() const noexcept {
    return fb_ext_[0];
  }

  char forwardExt() const noexcept {
    return fb_ext_[1];
  }

  auto fb_ext() const noexcept {
    return std::string(fb_ext_, 2);
  }

  auto next_kmer() const {
    return kmer_.forwardBase(fb_ext_[1]);
  }

  auto last_kmer() const {
    return kmer_.backwardBase(fb_ext_[0]);
  }
};

struct pkmer_t {
  unsigned char data[PACKED_KMER_LEN];

  pkmer_t(const std::string &kmer) {
    packKmer(kmer.data(), data);
  }

  pkmer_t(const pkmer_t &pkmer) {
    init(pkmer.data);
  }

  pkmer_t() {}

  std::string get() const noexcept {
    char kmer[KMER_LEN];
    unpackKmer(data, kmer);
    return std::string(kmer, KMER_LEN);
  }

  pkmer_t operator=(const pkmer_t &pkmer) {
    init(pkmer.data);
    return *this;
  }

  bool operator==(const pkmer_t &pkmer) const noexcept {
    return memcmp(pkmer.data, data, PACKED_KMER_LEN) == 0;
  }

  bool operator!=(const pkmer_t &pkmer) const noexcept {
    return !(*this == pkmer);
  }

  void init(const unsigned char data[PACKED_KMER_LEN]) {
    for (int i = 0; i < PACKED_KMER_LEN; i++) {
      this->data[i] = data[i];
    }
  }

  uint64_t hash() const noexcept {
    return MurmurHash3_x64_64(data, PACKED_KMER_LEN);
  }
};

struct kmer_pair {
  pkmer_t kmer;
  char fb_ext[2];
  int used = 0;

  kmer_pair() {}
  ~kmer_pair() {}

  kmer_pair(const std::string &kmer, const std::string &fb_ext) {
    init(kmer, fb_ext);
  }

  kmer_pair(const kmer_pair &kmer) {
    init(kmer);
  }

  kmer_pair operator=(const kmer_pair &kmer) {
    init(kmer);
    return *this;
  }

  uint64_t hash() const noexcept {
    return kmer.hash();
  }

  void init(const std::string &kmer, const std::string &fb_ext) {
    if (kmer.length() != KMER_LEN || fb_ext.length() != 2) {
      fprintf(stderr, "error: tried to initialize a kmer pair with too short a string.\n");
      return;
    }
    this->kmer = pkmer_t(kmer);
    for (int i = 0; i < 2; i++) {
      this->fb_ext[i] = fb_ext[i];
    }
  }

  void init(const kmer_pair &kmer) {
    this->kmer = kmer.kmer;
    for (int i = 0; i < 2; i++) {
      this->fb_ext[i] = kmer.fb_ext[i];
    }
    this->used = kmer.used;
  }

  bool operator==(const kmer_pair &kmer) const noexcept {
    return kmer.kmer == this->kmer && fb_ext[0] == kmer.fb_ext[0] &&
      fb_ext[1] == kmer.fb_ext[1];
  }

  bool operator!=(const kmer_pair &kmer) const noexcept {
    return !(kmer == *this);
  }

  pkmer_t next_kmer() const noexcept {
    return pkmer_t(kmer_str().substr(1, std::string::npos) + forwardExt());
  }

  pkmer_t last_kmer() const noexcept {
    return pkmer_t(backwardExt() + kmer_str().substr(0, kmer_str().length()-1));
  }

  char forwardExt() const noexcept {
    return fb_ext[1];
  }

  char backwardExt() const noexcept {
    return fb_ext[0];
  }

  std::string kmer_str() const noexcept {
    return kmer.get();
  }

  std::string fb_ext_str() const noexcept {
    return std::string(fb_ext, 2);
  }

  void print() const noexcept {
    printf("%s %s\n", kmer_str().c_str(), fb_ext_str().c_str());
  }
};
