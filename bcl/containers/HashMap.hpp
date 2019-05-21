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

namespace BCL {

// TODO: add KeyEqual, Allocator(?)

template <
  typename Key,
  typename T,
  typename Hash = std::hash<Key>,
  typename KeySerialize = BCL::serialize <Key>,
  typename ValSerialize = BCL::serialize <T>
  >
class HashMap {
public:

  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<const Key, T>;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using hasher = Hash;

  using HME = HashMapEntry <Key, T, KeySerialize, ValSerialize>;
  using KPTR = typename BCL::GlobalPtr <BCL::Container <Key, KeySerialize>>;
  using VPTR = typename BCL::GlobalPtr <BCL::Container <T, ValSerialize>>;

  constexpr static int free_flag = 0;
  constexpr static int reserved_flag = 1;
  constexpr static int ready_flag = 2;

  HashMap(const HashMap&) = delete;
  HashMap(HashMap&&) = delete;
  HashMap() = delete;

  // Initialize a HashMap of at least size size.
  HashMap(size_type capacity) : capacity_(capacity), team_ptr_(new BCL::WorldTeam()) {
    local_capacity_ = (capacity_ + BCL::nprocs(team()) - 1) / BCL::nprocs(team());
    hash_table_.resize(BCL::nprocs(team()), nullptr);

    hash_table_[BCL::rank(team())] = BCL::alloc <HME> (local_capacity_);

    if (hash_table_[BCL::rank(team())] == nullptr) {
      throw std::runtime_error("BCL::HashMap: ran out of memory\n");
    }

    HME* local_table = hash_table_[BCL::rank(team())].local();
    for (size_type i = 0; i < local_capacity_; i++) {
        new (&local_table[i]) HME();
    }

    for (size_type rank = 0; rank < BCL::nprocs(team()); rank++) {
      hash_table_[rank] = BCL::broadcast(hash_table_[rank], rank);
    }
    BCL::barrier();
  }

  HashMap(size_type capacity, const BCL::Team& team_) : capacity_(capacity), team_ptr_(team_.clone()) {
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

  ~HashMap() {
    if (!BCL::bcl_finalized) {
      if (team().in_team()) {
        if (BCL::rank(team()) < hash_table_.size() && hash_table_[BCL::rank(team())] != nullptr) {
          BCL::dealloc(hash_table_[BCL::rank(team())]);
        }
      }
    }
  }

  const BCL::Team& team() const {
    return *team_ptr_;
  }

  KPTR key_ptr(size_type slot) {
    return pointerto(key, slot_ptr(slot));
  }

  VPTR val_ptr(size_type slot) {
    return pointerto(val, slot_ptr(slot));
  }

  BlindHashMapReference<HashMap> operator[](const Key& key) {
    return BlindHashMapReference(this, key);
  }

  const BlindHashMapReference<HashMap> operator[](const Key& key) const {
    return BlindHashMapReference(this, key);
  }

  BlindHashMapIterator<HashMap> find(const Key& key,
                                     HashMapAL atomicity_level =
                                     HashMapAL::insert | HashMapAL::find) {
    return BlindHashMapIterator(this, key, atomicity_level);
  }

  GlobalHashMapIterator<HashMap> begin() {
    size_type slot = 0;
    while (!slot_ready(slot) && slot < capacity()) {
      slot++;
    }
    return GlobalHashMapIterator<HashMap>(this, slot);
  }

  GlobalHashMapIterator<HashMap> end() {
    return GlobalHashMapIterator<HashMap>(this, capacity());
  }

  LocalHashMapIterator<HashMap> local_begin() {
    size_type slot = 0;
    while (!local_slot_ready(slot) && slot < local_capacity()) {
      slot++;
    }
    return LocalHashMapIterator<HashMap>(this, slot);
  }

  LocalHashMapIterator<HashMap> local_end() {
    return LocalHashMapIterator<HashMap>(this, local_capacity());
  }

  // TODO: return iterator
  auto insert_or_assign(const key_type& k, const mapped_type& obj,
                        HashMapAL atomicity_level =
                        HashMapAL::insert | HashMapAL::find) {
    bool success;
    if (atomicity_level & HashMapAL::insert_find) {
      success = insert_atomic_impl_(k, obj);
    } else if (atomicity_level == HashMapAL::none){
      success = local_nonatomic_insert(HME{k, obj});
    }
    return std::make_pair(false, success);
  }

  bool insert_atomic_impl_(const Key &key, const T &val) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      success = request_slot(slot, key);
      if (success) {
        HME entry = get_entry(slot);
        entry.set_key(key);
        entry.set_val(val);
        set_entry(slot, entry);
        ready_slot(slot);
      }
    } while (!success && probe < capacity());
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
        entry.set_key(key);
        entry.set_val(fn(entry.get_val()));
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
    HME entry;
    int status;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      entry = atomic_get_entry(slot);
      status = entry.used;
      if (status & ready_flag) {
        success = (entry.get_key() == key);
      }
    } while (!success && !(status & free_flag) && probe < capacity());
    if (success) {
      val = entry.get_val();
      return true;
    } else {
      return false;
    }
  }

  bool find_or_insert(const Key &key, T &val) {
    size_t hash = hash_fn_(key);
    size_type probe = 0;
    bool success = false;
    HME entry;
    int status;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      entry = atomic_get_entry(slot);
      status = entry.used;
      if (status & ready_flag) {
        success = (entry.get_key() == key);
      }
    } while (!success && !(status & free_flag) && probe < capacity());
    if (success) {
      val = entry.get_val();
      return true;
    } else {
      return false;
    }
  }

  auto arfind(const Key& key) {
    return std::move(BCL::HMF<decltype(*this)>(key, *this));
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
      val = entry.get_val();
      return true;
    } else {
      return false;
    }
  }

  // Nonatomic with respect to remote inserts!
  bool local_nonatomic_insert(const HME &entry) {
    size_t hash = hash_fn_(entry.get_key());
    size_type probe = 0;
    bool success = false;
    auto* my_segment_ptr = hash_table_[BCL::rank(team())].local();
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      size_type node_slot = slot - (local_capacity_*BCL::rank(team()));
      if (node_slot >= local_capacity_ || node_slot < 0) {
        break;
      } else if (my_segment_ptr[node_slot].used == free_flag) {
        my_segment_ptr[node_slot].used = ready_flag;
        my_segment_ptr[node_slot].set_key(entry.get_key());
        my_segment_ptr[node_slot].set_val(entry.get_val());
        success = true;
      } else if (my_segment_ptr[node_slot].used == ready_flag &&
                 my_segment_ptr[node_slot].get_key() == entry.get_key()) {
        my_segment_ptr[node_slot].set_val(entry.get_val());
        success = true;
      }
    } while (!success);
    return success;
  }

  HME get_entry(size_type slot) {
    return BCL::rget(slot_ptr(slot));
  }

  future<HME> arget_entry(size_type slot) {
    return std::move(BCL::arget(slot_ptr(slot)));
  }

  // Upon return, read bit in slot is set.
  HME atomic_get_entry(size_type slot) {
    int old_value = ready_flag;

    // Bits 2 -> 32 are for readers to mark reserved
    int read_bit = 2 + (lrand48() % 30);

    auto ptr = slot_used_ptr(slot);

    int return_value = BCL::fetch_and_op<int>(ptr, (0x1 << read_bit), BCL::or_<int>{});

    if (return_value & ready_flag && !(return_value & (0x1 << read_bit))) {
      auto entry = get_entry(slot);
      int rv = BCL::fetch_and_op<int>(ptr, ~(0x1 << read_bit), BCL::and_<int>{});
      return entry;
    } else if ((return_value & (0x3)) == 0) {
      if (!(return_value & (0x1 << read_bit))) {
        int rv = BCL::fetch_and_op<int>(ptr, ~(0x1 << read_bit), BCL::and_<int>{});
      }
      return HME{};
    } else {
      // OPTIMIZE: use return value to pick a reader bit "hint"
      //           unclear if this will help, since reader bits
      //           will only ever be flipped for around 4-6us.
      if (!(return_value & (0x1 << read_bit))) {
        int rv = BCL::fetch_and_op<int>(ptr, ~(0x1 << read_bit), BCL::and_<int>{});
      }
      return atomic_get_entry(slot);
    }
  }

  void set_entry(size_type slot, const HME &entry) {
    if (slot >= capacity()) {
      throw std::runtime_error("slot too large!!!");
    }
    BCL::memcpy(slot_ptr(slot), &entry, offsetof(HME, used));
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
        int xor_value = 0x3;
        BCL::fetch_and_op<int>(slot_used_ptr(slot), xor_value, BCL::xor_<int>{});
        return false;
      }
    } else {
      return true;
    }
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

  BCL::GlobalPtr<HME> slot_ptr(size_t slot) {
    size_t node = slot / local_capacity_;
    size_t node_slot = slot - node*local_capacity_;
    return hash_table_[node] + node_slot;
  }

  BCL::GlobalPtr<int> slot_used_ptr(size_t slot) {
    return pointerto(used, slot_ptr(slot));
  }

  size_type capacity() const noexcept {
    return capacity_;
  }

  size_type local_capacity() const noexcept {
    return local_capacity_;
  }

  void ready_slot(size_type slot) {
    int xor_value = 0x3;
    int val = BCL::fetch_and_op<int>(slot_used_ptr(slot), xor_value, BCL::xor_<int>{});
    assert(val & reserved_flag);
  }

  size_type get_probe(size_type probe) {
    return probe*probe;
  }

  size_type capacity_;
  size_type local_capacity_;

  std::unique_ptr<BCL::Team> team_ptr_;

  Hash hash_fn_;

  std::vector <BCL::GlobalPtr <HME>> hash_table_;
};

} // end BCL
