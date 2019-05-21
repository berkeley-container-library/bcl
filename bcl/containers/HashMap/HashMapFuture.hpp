#pragma once

#include <bcl/core/future.hpp>
#include <optional>

// TODO: eliminate success_ by making rv_ std::optional.

namespace BCL {

template <typename H>
class HashMapFuture {
public:
  using hash_type = std::remove_cv_t<std::remove_reference_t<H>>;
  using key_type = typename hash_type::key_type;
  using mapped_type = typename hash_type::mapped_type;
  using HME = typename hash_type::HME;

  HashMapFuture(HashMapFuture&&) = default;
  HashMapFuture& operator=(HashMapFuture&&) = default;
  HashMapFuture(const HashMapFuture&) = delete;
  HashMapFuture& operator=(const HashMapFuture&) = delete;

  HashMapFuture(const key_type& key, H& hash_map) : key_(key), hash_map_(hash_map) {
    hash_ = hash_map_.hash_fn(key);

    uint64_t slot = (hash_ + hash_map_.get_probe(probe_++)) % hash_map_.size;
    entry_ = std::move(hash_map_.arget_entry(slot));
  }

  template <class Rep, class Period>
  std::future_status wait_for(const std::chrono::duration<Rep,Period>& timeout_duration) {
    if (success_) {
      return std::future_status::ready;
    }
    if (entry_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return std::future_status::timeout;
    } else {
      HME entry = entry_.get();
      int status = entry.used;
      if (status == hash_map_.free_flag) {
        success_ = true;
        value_ = entry;
        return std::future_status::ready;
      } else {
        if (entry.get_key() == key_) {
          success_ = true;
          value_ = entry;
          return std::future_status::ready;
        } else {
          uint64_t slot = (hash_ + hash_map_.get_probe(probe_++)) % hash_map_.size;
          entry_ = std::move(hash_map_.arget_entry(slot));
          return std::future_status::timeout;
        }
      }
    }
  }

  std::optional<mapped_type> get() {
    size_t count = 0;
    while (!success_ && wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      entry_.wait();
    }

    if (value_.used == hash_map_.ready_flag && value_.get_key() == key_) {
      return value_.get_val();
    } else {
      return {};
    }
  }

private:
  uint64_t hash_;
  uint64_t probe_ = 0;
  bool success_ = false;
  BCL::future<HME> entry_;
  HME value_;
  H& hash_map_;
  key_type key_;
};

template <typename T>
using HMF = HashMapFuture<T>;

} // end BCL
