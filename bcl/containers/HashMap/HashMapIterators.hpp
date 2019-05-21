#pragma once

#include <bcl/bcl.hpp>

namespace BCL {

template <typename T>
class GlobalHashMapIterator;

// "Blindly" (only by key, not by slot)
// reference a HashMap entry
template <typename T>
class BlindHashMapReference {
public:

  using key_type = typename T::key_type;
  using mapped_type = typename T::mapped_type;

  BlindHashMapReference(T* hashmap, const key_type& key,
                        HashMapAL atomicity_level = HashMapAL::insert_find)
                          : hashmap_(hashmap), key_(key),
                            atomicity_level_(atomicity_level){}

  // XXX: return reference to self?
  mapped_type operator=(const mapped_type& value) {
    bool success = hashmap_->insert_atomic_impl_(key_, value);
    assert(success);
    return value;
  }

  operator mapped_type() const {
    mapped_type value;
    bool success;
    if (atomicity_level_ <= HashMapAL::find) {
      success = hashmap_->find_nonatomic_impl_(key_, value);
    } else {
      success = hashmap_->find_atomic_impl_(key_, value);
    }
    return value;
  }

private:
  key_type key_;
  T* hashmap_;
  HashMapAL atomicity_level_;
};

template <typename T>
class BlindHashMapIterator {
public:

  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using mapped_type = typename T::mapped_type;

  BlindHashMapIterator(T* hashmap, const key_type& key,
                       HashMapAL atomicity_level = HashMapAL::insert_find)
                         : hashmap_(hashmap), key_(key),
                           atomicity_level_(atomicity_level) {}

  BlindHashMapReference<T> operator*() const {
    return BlindHashMapReference<T>(hashmap_, key_, atomicity_level_);
  }

  bool operator==(const GlobalHashMapIterator<T>& iter) {
    BCL_DEBUG(
      if (iter != hashmap_->end()) {
        throw debug_error("Compared a BlindHashMapIterator with a non-end GlobalHashMapIterator.");
      }
    )

    mapped_type value;
    bool success;
    if (atomicity_level_ <= HashMapAL::find) {
      success = hashmap_->find_nonatomic_impl_(key_, value);
    } else {
      success = hashmap_->find_atomic_impl_(key_, value);
    }
    return !success;
  }

  bool operator!=(const GlobalHashMapIterator<T>& iter) {
    return !(*this == iter);
  }

private:
  key_type key_;
  HashMapAL atomicity_level_;
  T* hashmap_;
};

template <typename T>
class GlobalHashMapReference {
public:
  using size_type = typename T::size_type;
  using difference_type = typename T::difference_type;

  using key_type = typename T::key_type;
  using mapped_type = typename T::mapped_type;
  using value_type = typename T::value_type;

  GlobalHashMapReference(T* hashmap, size_type slot) : hashmap_(hashmap), slot_(slot) {}

  operator value_type() const {
    auto entry = hashmap_->get_entry(slot_);
    assert(entry.used == hashmap_->ready_flag);

    return value_type{entry.get_key(), entry.get_val()};
  }

  value_type operator=(const value_type& value) {
    typename T::HME entry;
    entry.used = hashmap_->ready_flag;
    entry.set_key(value.first);
    entry.set_val(value.second);
    BCL::memcpy(hashmap_->slot_ptr(slot_), &entry, sizeof(entry));
    // XXX: to flush or not?
    return value;
  }

private:
  T* hashmap_;
  size_type slot_;
};

template <typename T>
class GlobalHashMapIterator {
public:
  using size_type = typename T::size_type;
  using difference_type = typename T::difference_type;

  using key_type = typename T::key_type;
  using mapped_type = typename T::mapped_type;
  using value_type = typename T::value_type;

  using reference = GlobalHashMapReference<T>;

  GlobalHashMapIterator(T* hashmap, size_type slot) : hashmap_(hashmap), slot_(slot) {}

  bool operator==(const GlobalHashMapIterator& other) const noexcept {
    return hashmap_ == other.hashmap_ && slot_ == other.slot_;
  }

  bool operator!=(const GlobalHashMapIterator& other) const noexcept {
    return !(*this == other);
  }

  GlobalHashMapIterator& operator++() {
    size_type new_slot = std::min(slot_+1, hashmap_->capacity_);
    while (new_slot < hashmap_->capacity_ && !hashmap_->slot_ready(new_slot)) {
      new_slot++;
    }
    slot_ = new_slot;
    return *this;
  }

  GlobalHashMapIterator operator++(int) {
    size_type old_slot = slot_;
    size_type new_slot = std::min(slot_+1, hashmap_->capacity_);
    while (new_slot < hashmap_->capacity_ && !hashmap_->slot_ready(new_slot)) {
      new_slot++;
    }
    slot_ = new_slot;
    return GlobalHashMapIterator(hashmap_, old_slot);
  }

  GlobalHashMapReference<T> operator*() {
    assert(slot_ < hashmap_->capacity_);
    return GlobalHashMapReference<T>(hashmap_, slot_);
  }

private:
  T* hashmap_;
  size_type slot_;
};

template <typename HashMap>
class LocalHashMapReference {
public:
  using size_type = typename HashMap::size_type;
  using difference_type = typename HashMap::difference_type;

  using key_type = typename HashMap::key_type;
  using mapped_type = typename HashMap::mapped_type;
  using value_type = typename HashMap::value_type;

  LocalHashMapReference() = delete;
  LocalHashMapReference(const LocalHashMapReference&) = default;
  LocalHashMapReference& operator=(const LocalHashMapReference&) = default;

  LocalHashMapReference(HashMap* hashmap, size_type idx) : hashmap_(hashmap), idx_(idx) {}

  operator value_type() const {
    return value_type(hashmap_->hash_table_[BCL::rank()].local()[idx_].get_key(),
                      hashmap_->hash_table_[BCL::rank()].local()[idx_].get_val());
  }

  value_type operator=(const value_type& value) const {
    hashmap_->hash_table_[BCL::rank()].local()[idx_].set_key(std::get<0>(value));
    hashmap_->hash_table_[BCL::rank()].local()[idx_].set_val(std::get<1>(value));
    return value;
  }

private:
  size_t idx_;
  HashMap* hashmap_;
};

template <typename HashMap>
class LocalHashMapIterator {
public:
  using size_type = typename HashMap::size_type;
  using difference_type = typename HashMap::difference_type;

  using key_type = typename HashMap::key_type;
  using mapped_type = typename HashMap::mapped_type;
  using value_type = typename HashMap::value_type;

  LocalHashMapIterator(HashMap* hashmap, size_type idx) : hashmap_(hashmap), idx_(idx) {}

  LocalHashMapReference<HashMap> operator*() {
    return LocalHashMapReference<HashMap>(hashmap_, idx_);
  }

  LocalHashMapIterator& operator++() {
    size_type new_slot = std::min(idx_+1, hashmap_->local_capacity());
    while (new_slot < hashmap_->local_capacity() && !hashmap_->local_slot_ready(new_slot)) {
      new_slot++;
    }
    idx_ = new_slot;
    return *this;
  }

  LocalHashMapIterator operator++(int) {
    size_type old_slot = idx_;
    size_type new_slot = std::min(idx_+1, hashmap_->local_capacity());
    while (new_slot < hashmap_->local_capacity() && !hashmap_->local_slot_ready(new_slot)) {
      new_slot++;
    }
    idx_ = new_slot;
    return LocalHashMapIterator(hashmap_, old_slot);
  }

  bool operator==(const LocalHashMapIterator& other) {
    return hashmap_ == other.hashmap_ && idx_ == other.idx_;
  }

  bool operator!=(const LocalHashMapIterator& other) {
    return !(*this == other);
  }

private:
  size_t idx_;
  HashMap* hashmap_;
};

} // end BCL
