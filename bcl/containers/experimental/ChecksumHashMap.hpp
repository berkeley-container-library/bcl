#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <type_traits>
#include <atomic>
#include <cassert>

#include <bcl/bcl.hpp>
#include <bcl/containers/Container.hpp>
#include <bcl/containers/HashMap/HashMapAL.hpp>
#include <bcl/containers/HashMap/HashMapEntry.hpp>
#include <bcl/containers/HashMap/HashMapFuture.hpp>
#include <bcl/containers/HashMap/HashMapIterators.hpp>
#include <bcl/core/util/Backoff.hpp>

namespace BCL {

template <typename T>
struct HashedData {
  T data;
  size_t hash;
};

template <
  typename Key,
  typename T,
  typename Hash = std::hash<Key>,
  typename Checksum = std::hash<T>,
  typename KeySerialize = BCL::serialize <Key>,
  typename HashedValSerialize = BCL::serialize <HashedData<T>>
  >
class ChecksumHashMap {
public:

  using key_type = Key;
  using mapped_type = HashedData<T>;
  using value_type = std::pair<const key_type, mapped_type>;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using hasher = Hash;

  using HME = HashMapEntry <key_type, mapped_type, KeySerialize, HashedValSerialize>;
  using KPTR = typename BCL::GlobalPtr <BCL::Container <key_type, KeySerialize>>;
  using VPTR = typename BCL::GlobalPtr <BCL::Container <mapped_type, HashedValSerialize>>;

  constexpr static int free_flag = 0;
  constexpr static int reserved_flag = 1;
  constexpr static int ready_flag = 2;

private:
    size_type capacity_;
    size_type local_capacity_;
    std::unique_ptr<BCL::Team> team_ptr_;
    Hash hash_fn_;
    Checksum checksum_fn_;
    std::vector <BCL::GlobalPtr <HME>> hash_table_;

public:
  ChecksumHashMap(const ChecksumHashMap&) = delete;
  ChecksumHashMap(ChecksumHashMap&&) = delete;
  ChecksumHashMap() = delete;
  ChecksumHashMap &operator=(const ChecksumHashMap&) = delete;
  ChecksumHashMap &operator=(ChecksumHashMap&&) = delete;

  // Initialize a HashMap of at least size size.
  explicit ChecksumHashMap(size_type capacity, const BCL::Team& team_ = BCL::WorldTeam()) : capacity_(capacity), team_ptr_(team_.clone()) {
    local_capacity_ = (capacity_ + BCL::nprocs(team()) - 1) / BCL::nprocs(team());
    hash_table_.resize(BCL::nprocs(team()), nullptr);

    if (team().in_team()) {
      hash_table_[BCL::rank(team())] = BCL::alloc <HME> (local_capacity_);

      if (hash_table_[BCL::rank(team())] == nullptr) {
        throw std::runtime_error("BCL::HashMap: ran out of memory\n");
      }

      HME* local_table = hash_table_[BCL::rank(team())].local();
      for (size_type i = 0; i < local_capacity_; i++) {
        new (&local_table[i]) HME();
      }
    }

    for (size_type rank = 0; rank < hash_table_.size(); rank++) {
      hash_table_[rank] = BCL::broadcast(hash_table_[rank], team().to_world(rank));
    }
  }

  ~ChecksumHashMap() {
      //if bcl_finalized, the shared memory has been released.
    if (!BCL::bcl_finalized) {
      if (team().in_team()) {
        if (BCL::rank(team()) < hash_table_.size() && hash_table_[BCL::rank(team())] != nullptr) {
          BCL::dealloc(hash_table_[BCL::rank(team())]);
        }
      }
    }
  }

  // TODO: return iterator
  bool insert(const key_type& k, const T& obj,
                        HashMapAL atomicity_level =
                        HashMapAL::insert | HashMapAL::find) {
    bool success = false;
    if (atomicity_level & HashMapAL::insert) {
      success = insert_atomic_impl_(k, obj);
    } else {
      success = insert_nonatomic_impl_(k, obj);
    }
    return success;
  }

  // find value
  bool find_value(const key_type& k, T &obj, HashMapAL atomicity_level =
              HashMapAL::insert | HashMapAL::find) {
    bool success = false;
    if (atomicity_level & HashMapAL::insert) {
      success = find_atomic_impl_(k, obj);
    } else {
      success = find_nonatomic_impl_(k, obj);
    }
    return success;
  }

  BlindHashMapIterator<ChecksumHashMap> find(const Key& key,
                                     HashMapAL atomicity_level =
                                     HashMapAL::insert | HashMapAL::find) {
    return BlindHashMapIterator<ChecksumHashMap>(this, key, atomicity_level);
  }

  BlindHashMapReference<ChecksumHashMap> operator[](const Key& key) {
    return BlindHashMapReference<ChecksumHashMap>(this, key);
  }

  const BlindHashMapReference<ChecksumHashMap> operator[](const Key& key) const {
    return BlindHashMapReference<ChecksumHashMap>(this, key);
  }

  GlobalHashMapIterator<ChecksumHashMap> begin() {
    size_type slot = 0;
    while (!slot_ready(slot) && slot < capacity()) {
      slot++;
    }
    return GlobalHashMapIterator<ChecksumHashMap>(this, slot);
  }

  GlobalHashMapIterator<ChecksumHashMap> end() {
    return GlobalHashMapIterator<ChecksumHashMap>(this, capacity());
  }

  LocalHashMapIterator<ChecksumHashMap> local_begin() {
    size_type slot = 0;
    while (!local_slot_ready(slot) && slot < local_capacity()) {
      slot++;
    }
    return LocalHashMapIterator<ChecksumHashMap>(this, slot);
  }

  LocalHashMapIterator<ChecksumHashMap> local_end() {
    return LocalHashMapIterator<ChecksumHashMap>(this, local_capacity());
  }

  const BCL::Team& team() const {
    return *team_ptr_;
  }

  size_type capacity() const noexcept {
    return capacity_;
  }

  size_type local_capacity() const noexcept {
    return local_capacity_;
  }

  auto arfind(const Key& key) {
    return std::move(BCL::HMF<decltype(*this)>(key, *this));
  }

  future<HME> arget_entry(size_type slot) {
    return std::move(BCL::arget(slot_ptr(slot)));
  }

private:
  size_t get_hash(const T& val, int loc) {
    size_t hashed = checksum_fn_(val);
    bool sense = (loc / capacity()) % 2;
    // This occurs if you add an int=0. This is problematic since the
    // default data is zero, so processors can pop unwritten data if zero hashes to zero.
    // TODO preferably a more robust method of dealing with this...
    // XXX: I believe the best solution is just to use a more robust hash fn.
    //      There are some options in /bcl/core/detail/hash_functions,
    //      and I'll try to add some more.
    if (hashed == 0) {
      hashed = 42;
    }
    if (sense) {
      hashed = ~hashed;
    }
    return hashed;
  }

  // insert related methods
  bool insert_atomic_impl_(const Key &key, const T &val) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      success = request_slot(slot, key);
      if (success) {
        HME entry(key, HashedData<T>{val, get_hash(val, slot)});
        set_entry(slot, entry);
        ready_slot(slot);
      }
    } while (!success && probe < capacity());
    return success;
  }

    // Nonatomic with respect to remote inserts!
  bool insert_nonatomic_impl_(const Key &key, const T &val) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    auto* my_segment_ptr = hash_table_[BCL::rank(team())].local();
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      size_type node_slot = slot - (local_capacity_*BCL::rank(team()));
      if (node_slot >= local_capacity_ || node_slot < 0) {
        break;
      } else if (my_segment_ptr[node_slot].used == free_flag) {
        my_segment_ptr[node_slot].set_key(key);
        my_segment_ptr[node_slot].set_val(HashedData<T>{val, get_hash(val, slot)});
        success = true;
      } else if (my_segment_ptr[node_slot].used == ready_flag &&
                 my_segment_ptr[node_slot].get_key() == key) {
        my_segment_ptr[node_slot].set_val(HashedData<T>{val, get_hash(val, slot)});
        success = true;
      }
    } while (!success);
    return success;
  }

  template <typename Fn>
  bool modify(const Key& key, Fn&& fn) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      success = request_slot(slot, key);
      if (success) {
        HME entry = get_entry(slot);
        T new_val = fn(entry.get_val().val);
        entry.set_val(HashedData<T>(new_val, get_hash(new_val, slot)));
        set_entry(slot, entry);
        ready_slot(slot);
      }
    } while (!success && probe < capacity());
    return success;
  }

  bool find_atomic_impl_(const Key &key, T &val) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    size_type slot;
    HME entry;
    int status;
    do {
      slot = (hash + get_probe(probe++)) % capacity();
      entry = get_entry(slot);
      status = entry.used;
      if (status == ready_flag) {
        success = (entry.get_key() == key);
      }
    } while (!success && status != free_flag && probe < capacity());
    if (success) {
      Backoff backoff;
      while(true) {
        HashedData<T> hashedVal = entry.get_val();
        if (hashedVal.hash == get_hash(hashedVal.data, slot)) {
          val = hashedVal.data;
          return true;
        } else {
          backoff.backoff();
          backoff.increase_backoff_impl_();
          entry = get_entry(slot);
        }
      }
    } else {
      return false;
    }
  }

  bool find_nonatomic_impl_(const Key &key, T &val) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    HME entry;
    int status;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      entry = get_entry(slot);
      status = entry.used;
      if (status == ready_flag) {
        success = (entry.get_key() == key);
      }
    } while (!success && status != free_flag && probe < capacity());
    if (success) {
      val = entry.get_val().data;
      return true;
    } else {
      return false;
    }
  }

  /*
   Request slot for key.
   free_flag -> reserved_flag
   ready_flag -> reserved_flag
  */
  bool request_slot(size_type slot, const Key& key) {
    bool success = false;
    int current_value = free_flag;
    int return_value;

    do {
      return_value = BCL::int_compare_and_swap(slot_used_ptr(slot),
                                               current_value, reserved_flag);
      if (current_value == return_value) {
        success = true;
      }

      if (return_value & ready_flag || return_value & reserved_flag) {
        current_value = ready_flag;
      }
    } while (!success);

    // At this point, you have slot -> reserved_flag

    if (return_value & ready_flag) {
      HME entry = get_entry(slot);
      if (entry.get_key() == key) {
        return true;
      } else {
        // get wrong slot
        int xor_value = 0x3;
        BCL::fetch_and_op<int>(slot_used_ptr(slot), xor_value, BCL::xor_<int>{});
        return false;
      }
    } else {
      // empty
      return true;
    }
  }

  HME get_entry(size_type slot) {
    if (slot >= capacity()) {
      throw std::runtime_error("slot too large!!!");
    }
    return BCL::rget(slot_ptr(slot));
  }

  void set_entry(size_type slot, const HME &entry) {
    if (slot >= capacity()) {
      throw std::runtime_error("slot too large!!!");
    }
    BCL::memcpy(slot_ptr(slot), &entry, offsetof(HME, used));
  }

  BCL::GlobalPtr<HME> slot_ptr(size_t slot) {
    size_t node = slot / local_capacity_;
    size_t node_slot = slot - node*local_capacity_;
    return hash_table_[node] + node_slot;
  }

  BCL::GlobalPtr<int> slot_used_ptr(size_t slot) {
    return pointerto(used, slot_ptr(slot));
  }

  int slot_status(size_type slot) {
    return BCL::rget_atomic(slot_used_ptr(slot));
  }

  bool slot_ready(size_type slot) {
    return slot_status(slot) == ready_flag;
  }

  bool local_slot_ready(size_type slot) {
    return hash_table_[BCL::rank()].local()[slot].used == ready_flag;
  }

  void ready_slot(size_type slot) {
    int xor_value = 0x3;
    int val = BCL::fetch_and_op<int>(slot_used_ptr(slot), xor_value, BCL::xor_<int>{});
    assert(val & reserved_flag);
  }

  size_type get_probe(size_type probe) {
    return probe*probe;
  }

  KPTR key_ptr(size_type slot) {
    return pointerto(key, slot_ptr(slot));
  }

  VPTR val_ptr(size_type slot) {
    return pointerto(val, slot_ptr(slot));
  }
};

} // end BCL
