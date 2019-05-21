#pragma once

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ListQueue.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <bcl/containers/CachedCopy.hpp>

namespace BCL {

template <typename T>
struct SlabQueue;

template <typename T>
struct SlabQueueIterator {
  typename BCL::FastQueue<T>::iterator loc_;
  BCL::QueueIterator<BCL::FastQueue<T>> slab_;
  SlabQueue<T>* queue_ptr_ = nullptr;

  SlabQueueIterator(BCL::QueueIterator<BCL::FastQueue<T>> slab,
                    typename BCL::FastQueue<T>::iterator loc,
                    SlabQueue<T>& queue) : loc_(loc), slab_(slab), queue_ptr_(&queue) {
  }

  auto operator++() {
    loc_++;
    if (loc_ == (*slab_).end()) {
      slab_++;
      if (slab_ == queue_ptr_->queue_.end()) {
        loc_ = nullptr;
      } else {
        loc_ = (*slab_).begin();
      }
    }
    return *this;
  }

  auto operator++(int) {
    loc_++;
    if (loc_ == (*slab_).end()) {
      slab_++;
      if (slab_ == queue_ptr_->queue_.end()) {
        loc_ = nullptr;
      } else {
        loc_ = (*slab_).begin();
      }
    }
    return *this;
  }

  auto operator*() {
    return *loc_;
  }

  bool operator==(const SlabQueueIterator<T>& other) {
    return slab_ == other.slab_ && loc_ == other.loc_;
  }

  bool operator!=(const SlabQueueIterator<T>& other) {
    return !(*this == other);
  }

  SlabQueueIterator(const SlabQueueIterator&) = default;
  SlabQueueIterator() {}
};

template <typename T>
struct SlabQueue {
  BCL::ListQueue<BCL::FastQueue<T>> queue_;
  size_t slab_size_;
  size_t host_;

  BCL::CachedCopy<BCL::FastQueue<T>> tail_buf_;
  BCL::QueueIterator<BCL::FastQueue<T>> tail_buf_iter_;

  SlabQueue(size_t host, size_t slab_size) :
            queue_(host), host_(host), slab_size_(slab_size) {
    if (BCL::rank() == host) {
      queue_.push(std::move(BCL::FastQueue<T>(slab_size)));
    }
    BCL::barrier();
    tail_buf_iter_ = queue_.back_iter();
    tail_buf_ = BCL::CachedCopy(&*tail_buf_iter_);
  }

  void push(const T& value) {
    bool success = tail_buf_->push(value);
    if (!success) {
      auto queue = BCL::FastQueue<T>(slab_size_);
      // bool success = queue_.push_after(std::move(queue), tail_buf_iter_);
      bool success = queue_.emplace_after(tail_buf_iter_, slab_size_);
      tail_buf_iter_ = queue_.back_iter();
      tail_buf_ = BCL::CachedCopy(&*tail_buf_iter_);
      push(value);
    }
  }

  auto begin() {
    return SlabQueueIterator<T>(queue_.begin(), (*queue_.begin()).begin(), *this);
  }

  auto end() {
    return SlabQueueIterator<T>(queue_.end(), nullptr, *this);
  }
};

}
