#pragma once

namespace BCL {

// TODO: complete this

template <
          typename T,
          typename Allocator = std::allocator<T>
          >
class vector {
public:
  using allocator_type = Allocator;
  using size_type = typename Allocator::size_type;
  using pointer = typename Allocator::pointer;
  using const_pointer = typename Allocator::const_pointer;
  using reference = typename Allocator::reference;
  using const_reference = typename Allocator::const_reference;
  using difference_type = typename Allocator::difference_type;

  vector() = default;

  // TODO: implement
  vector(const vector& other) = delete;

  vector(vector&& other) :
    allocator_(std::move(other.allocator_)),
    ptr_(std::move(other.ptr_)),
    size_(std::move(other.size_)),
    capacity_(std::move(other.capacity_)) {}

  vector(size_type count) {
    ptr_ = allocator_.allocate(count);
    capacity_ = count;
    size_ = count;

    for (size_t i = 0; i < size(); i++) {
      allocator_.construct(ptr_ + i);
    }
  }

  vector& operator=(vector&& other) {
    allocator_ = std::move(other.allocator_);
    ptr_ = std::move(other.ptr_);
    size_ = std::move(other.size_);
    capacity_ = std::move(other.capacity_);
    return *this;
  }

  void reserve(size_t new_cap) {
    if (new_cap > capacity()) {
      pointer new_ptr = allocator_.allocate(new_cap);
      capacity_ = new_cap;

      if (size() > 0) {
        for (size_t i = 0; i < size(); i++) {
          new_ptr[i] = ptr_[i];
        }
      }
      std::swap(new_ptr, ptr_);
      if (new_ptr != nullptr) {
        allocator_.deallocate(new_ptr);
      }
    }
  }

  const_reference operator[](size_type idx) const {
    return ptr_[idx];
  }

  reference operator[](size_type idx) {
    return ptr_[idx];
  }

  pointer data() {
    return ptr_;
  }

  const_pointer data() const {
    return ptr_;
  }

  size_type size() const {
    return size_;
  }

  bool empty() const {
    return size() == 0;
  }

  size_type capacity() const {
    return capacity_;
  }

private:
  allocator_type allocator_;
  pointer ptr_ = nullptr;
  size_type size_ = 0;
  size_type capacity_ = 0;
};

}
