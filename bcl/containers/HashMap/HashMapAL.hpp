#pragma once

namespace BCL {

struct HashMapAL {
  constexpr static int none = 0x0;
  constexpr static int find = (0x1 << 0);
  constexpr static int insert = (0x1 << 1);
  constexpr static int insert_find = insert | find;
  constexpr static int modify = insert_find;

  int val;

  HashMapAL(int val) : val(val) {}
  HashMapAL& operator=(const HashMapAL&) = default;

  operator int() const {
    return val;
  }

  HashMapAL& operator=(int val) {
    this->val = val;
    return *this;
  }

  bool operator==(int val) const {
    return this->val == val;
  }
};

}
