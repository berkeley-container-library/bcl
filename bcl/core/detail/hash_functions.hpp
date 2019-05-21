#pragma once

namespace BCL {

template <typename T>
struct djb2_hash {
  unsigned long djb2(const unsigned char *str, std::size_t len) const noexcept {
    unsigned long hash = 5381;

    for (size_t i = 0; i < len; i++) {
      int c = str[i];
      hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
  }

  std::size_t operator()(const T& k) const noexcept {
    return djb2(reinterpret_cast<const unsigned char*>(&k), sizeof(T));
  }
};

template <typename T>
struct nagasaka_hash {
  std::size_t operator()(const T& k) const noexcept {
    return k*107;
  }
};

}
