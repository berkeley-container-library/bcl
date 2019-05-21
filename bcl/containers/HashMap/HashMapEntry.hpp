#pragma once

#include <bcl/bcl.hpp>
#include <bcl/containers/Container.hpp>

namespace BCL {

template <
  typename K,
  typename V,
  typename KeySerialize = BCL::serialize <K>,
  typename ValSerialize = BCL::serialize <V>
>
class HashMapEntry {
public:
  BCL::Container <K, KeySerialize> key;
  BCL::Container <V, ValSerialize> val;
  int used = 0;

  HashMapEntry(const K &key, const V &val) {
    insert(key, val);
  }

  HashMapEntry() = default;
  HashMapEntry(const HashMapEntry& entry) = default;
  HashMapEntry(HashMapEntry&& entry) = default;
  HashMapEntry& operator=(const HashMapEntry&) = default;
  HashMapEntry& operator=(HashMapEntry&&) = default;

  void insert(const K &key, const V &val) {
    this->key.set(key);
    this->val.set(val);
  }

  K get_key() const {
    return key.get();
  }

  V get_val() const {
    return val.get();
  }

  void set_key(const K &key) {
    this->key.set(key);
  }

  void set_val(const V &val) {
    this->val.set(val);
  }
};

template <
  typename Key,
  typename T,
  typename KeySerialize,
  typename ValSerialize
>
struct serialize <HashMapEntry <Key, T, KeySerialize, ValSerialize>>
  : public BCL::identity_serialize <HashMapEntry <Key, T, KeySerialize, ValSerialize>> {};

}
