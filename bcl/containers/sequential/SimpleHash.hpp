#pragma once

#include <iterator>
#include <vector>
#include <tuple>

#include <cassert>

namespace BCL {

template <typename SHT>
class SimpleHashIterator;

template <
          typename Key,
          typename T,
          typename Hash = std::hash<Key>,
          typename Allocator = std::allocator<std::pair<Key, T>>
          >
class SimpleHash {
public:

  using iterator = SimpleHashIterator<SimpleHash>;
  using value_type = std::pair<Key, T>;
  using key_type = Key;
  using mapped_type = T;

  using allocator_traits = std::allocator_traits<Allocator>;
  using BAllocator = typename allocator_traits:: template rebind_alloc<bool>;

  SimpleHash() {}

  SimpleHash(const SimpleHash&) = default;
  SimpleHash& operator=(const SimpleHash&) = default;

  iterator end() {
    return SimpleHashIterator<SimpleHash>(*this, capacity());
  }

  iterator begin() {
    return begin_;
  }

  const iterator begin() const {
    return begin_;
  }

  const iterator end() const {
    return SimpleHashIterator<SimpleHash>(*const_cast<SimpleHash*>(this), capacity());
  }

  void accumulate(const value_type& value) {
    const key_type& key = value.first;
    const mapped_type& value_ = value.second;

    size_t hash = Hash{}(key);
    size_t probe = 0;

    bool success;

    do {
      // size_t slot = (hash + get_probe(probe++)) % capacity();
      size_t slot = (hash + get_probe(probe++)) & (capacity()-1);
      success = !occupied_[slot] || (data_[slot].first == key);

      if (success) {
        if (occupied_[slot]) {
          data_[slot].second += value_;
        } else {
          size_++;
          occupied_[slot] = true;
          data_[slot] = {key, value_};
          if (slot < begin_.idx_) {
            begin_ = iterator(*this, slot);
          }
        }
      }

    } while (!success && probe < capacity());

    assert(success);
  }

  T& operator[](const key_type& key) {
    size_t hash = Hash{}(key);
    size_t probe = 0;

    bool success;
    size_t slot;

    do {
      // slot = (hash + get_probe(probe++)) % capacity();
      slot = (hash + get_probe(probe++)) & (capacity()-1);
      success = !occupied_[slot] || (data_[slot].first == key);

      if (success) {
        if (!occupied_[slot]) {
          if (size() >= capacity()) {
            fprintf(stderr, "AGH! Size is %lu >= %lu\n", size(), capacity());
            assert(false);
          }
          size_++;
          occupied_[slot] = true;
          data_[slot] = {key, mapped_type{}};
          if (slot < begin_.idx_) {
            begin_ = iterator(*this, slot);
          }
        }
      }

    } while (!success && probe < capacity());

    assert(success);
    return data_[slot].second;
  }

  bool empty() const noexcept {
    return size() == 0;
  }

  size_t size() const noexcept {
    return size_;
  }

  size_t capacity() const noexcept {
    return data_.size();
  }

  void reserve(size_t count) {
    // XXX: currently does not shrink--should we do this?
    // TODO: support reserve while occupied.
    assert(empty());
    if (count > capacity()) {
      size_t count_ = 16;
      while (count_ < count) {
        count_ <<= 1;
      }
      occupied_.resize(count_, false);
      data_.resize(count_);
      begin_ = end();
    }
  }

  void clear() {
    std::fill(occupied_.begin(), occupied_.end(), false);
    begin_ = end();
    size_ = 0;
  }

  size_t get_probe(size_t probe) const noexcept {
    return probe;
  }

  // TODO: make iterator friends, protect these.
  std::vector<bool, BAllocator> occupied_;
  std::vector<value_type, Allocator> data_;

private:
  iterator begin_ = end();
  size_t size_ = 0;
};

template <typename SHT>
class SimpleHashIterator : public std::forward_iterator_tag {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = typename SHT::value_type;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::forward_iterator_tag;

  SimpleHashIterator(SHT& table, size_t idx) : hash_table_(&table), idx_(idx) {}

  SimpleHashIterator(const SimpleHashIterator&) = default;
  SimpleHashIterator& operator=(const SimpleHashIterator&) = default;

  bool operator==(const SimpleHashIterator& other) const {
    return hash_table_ == other.hash_table_ && idx_ == other.idx_;
  }

  bool operator!=(const SimpleHashIterator& other) const {
    return !(*this == other);
  }

  SimpleHashIterator& operator++() {
    if (idx_ < hash_table_->capacity()) {
      idx_++;
    }
    while (idx_ < hash_table_->capacity() && !hash_table_->occupied_[idx_]) {
      idx_++;
    }
    return *this;
  }

  // TODO: Actually post-increment
  void operator++(int) {
    if (idx_ < hash_table_->capacity()) {
      idx_++;
    }
    while (idx_ < hash_table_->capacity() && !hash_table_->occupied_[idx_]) {
      idx_++;
    }
  }

  typename SHT::value_type& operator*() {
    return hash_table_->data_[idx_];
  }

  const typename SHT::value_type& operator*() const {
    return hash_table_->data[idx_];
  }

  typename SHT::value_type* operator->() {
    return &hash_table_->data[idx_];
  }

  const typename SHT::value_type* operator->() const {
    return &hash_table_->data[idx_];
  }

  size_t idx_;
  SHT* hash_table_;
};

}
