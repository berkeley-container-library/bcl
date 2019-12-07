#pragma once

#include "packing.hpp"
#include "pkmer_t.hpp"

struct kmer_pair {
  pkmer_t kmer;
  char fb_ext[2];

  // Return the k-mer as a string
  std::string kmer_str() const noexcept;
  // Return the forward and backward extension as a string
  std::string fb_ext_str() const noexcept;

  // Return the next, previous kmer
  pkmer_t next_kmer() const noexcept;
  pkmer_t last_kmer() const noexcept;
  
  // Get the forward, backward extension.
  char forwardExt() const noexcept;
  char backwardExt() const noexcept;

  // Print the k-mer/fb_ext to STDOUT.
  void print() const noexcept;

  uint64_t hash() const noexcept;

  kmer_pair(const std::string &kmer, const std::string &fb_ext);

  kmer_pair() = default;
  ~kmer_pair() = default;

  kmer_pair(const kmer_pair &kmer) = default;
  kmer_pair& operator=(const kmer_pair &kmer) = default;

  void init(const std::string &kmer, const std::string &fb_ext);
  void init(const kmer_pair &kmer);

  bool operator==(const kmer_pair &kmer) const noexcept;
  bool operator!=(const kmer_pair &kmer) const noexcept;
};

char kmer_pair::forwardExt() const noexcept {
  return fb_ext[1];
}

char kmer_pair::backwardExt() const noexcept {
  return fb_ext[0];
}

std::string kmer_pair::kmer_str() const noexcept {
  return kmer.get();
}

std::string kmer_pair::fb_ext_str() const noexcept {
  return std::string(fb_ext, 2);
}

pkmer_t kmer_pair::next_kmer() const noexcept {
  return pkmer_t(kmer_str().substr(1, std::string::npos) + forwardExt());
}

pkmer_t kmer_pair::last_kmer() const noexcept {
  return pkmer_t(backwardExt() + kmer_str().substr(0, kmer_str().length()-1));
}

void kmer_pair::print() const noexcept {
  printf("%s %s\n", kmer_str().c_str(), fb_ext_str().c_str());
}

uint64_t kmer_pair::hash() const noexcept {
  return kmer.hash();
}

kmer_pair::kmer_pair(const std::string &kmer, const std::string &fb_ext) {
  init(kmer, fb_ext);
}

void kmer_pair::init(const std::string &kmer, const std::string &fb_ext) {
  if (kmer.length() != KMER_LEN || fb_ext.length() != 2) {
    fprintf(stderr, "error: tried to initialize a kmer pair with too short a string.\n");
    return;
  }
  this->kmer = pkmer_t(kmer);
  for (int i = 0; i < 2; i++) {
    this->fb_ext[i] = fb_ext[i];
  }
}

bool kmer_pair::operator==(const kmer_pair &kmer) const noexcept {
  return kmer.kmer == this->kmer && fb_ext[0] == kmer.fb_ext[0] &&
    fb_ext[1] == kmer.fb_ext[1];
}

bool kmer_pair::operator!=(const kmer_pair &kmer) const noexcept {
  return !(kmer == *this);
}
