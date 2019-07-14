#include "shm_segment.hpp"

namespace shm {

char* segment_ptr_ = nullptr;
std::string shm_key_;
size_t segment_size_;

void init_shm(std::string shm_key = "bcl_shmem", size_t size = 64*1024*1024) {
  segment_ptr_ = (char *) open_shared_segment(shm_key, size);
  shm_key_ = shm_key;
  segment_size_ = size;
}

void finalize_shm() {
  BCL::barrier();
  close_shared_segment(segment_ptr_, shm_key_, segment_size_);
}

size_t alignment_ = 64;
std::atomic<size_t>* heap_ptr_;
void init_basic_malloc() {
  heap_ptr_ = (std::atomic<size_t> *) segment_ptr_;
  BCL::barrier();
  if (BCL::rank() == 0) {
    new (heap_ptr_) std::atomic<size_t>;
    *heap_ptr_ = alignment_;
  }
  BCL::barrier();
}

void* basic_malloc(size_t n) {
  n = std::max(size_t(64), n);
  size_t aligned_size = alignment_*((n + alignment_ - 1) / alignment_);
  size_t offset = heap_ptr_->fetch_add(aligned_size);
  char* ptr = segment_ptr_ + offset;
  if (ptr + n <= segment_ptr_ + segment_size_) {
    return ptr;
  } else {
    heap_ptr_->fetch_add(-aligned_size);
    return nullptr;
  }
}

void basic_free(void* ptr) {}

// shm Allocator

template <typename T>
struct ptr;

template <typename T>
struct ref;

template <typename T>
struct allocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = ptr<T>;

  allocator() = default;

  template <typename U>
  allocator(const allocator<U>&) {}
  template <typename U>
  allocator& operator=(const allocator<U>&) {}

  template <typename U>
  allocator(allocator<U>&&) {}
  template <typename U>
  allocator& operator=(allocator<U>&&) {}

  pointer allocate(size_t n) const {
    char* ptr_ = (char *) basic_malloc(sizeof(T)*n);
    if (ptr_ == nullptr) {
      throw std::bad_alloc();
    }
    size_t offset = ptr_ - segment_ptr_;
    fprintf(stderr, "Returning %s (0x%p)\n", pointer(offset).str().c_str(), ptr_);
    return pointer(offset);
  }

  template <typename X, typename... Args>
  void construct(X* xptr, Args&&... args) {
    printf("(Construct %lu) Got pointer 0x%p (Base %p, offset %lu)\n",
           BCL::rank(), xptr, segment_ptr_, (char *) xptr - segment_ptr_);
    fflush(stdout);
    new (xptr) X(std::forward<Args>(args)...);
  }

  template <typename X>
  void destroy(X* xptr) {
    printf("(Destroy %lu) Got pointer 0x%p (Base %p, offset %lu)\n",
           BCL::rank(), xptr, segment_ptr_, (char *) xptr - segment_ptr_);
    fflush(stdout);
    delete xptr;
  }

  template <typename X, typename... Args>
  void construct(ptr<X> xptr, Args&&... args) {
    new (xptr.local()) X(std::forward<Args>(args)...);
  }

  template <typename X>
  void destroy(ptr<X> xptr) {
    delete xptr.local();
  }

  void deallocate(pointer p, size_t n) const {
    basic_free(p.local());
  }

  bool operator==(const allocator&) const {
    return true;
  }

  bool operator!=(const allocator& other) const {
    return !(*this == other);
  }
};

template <typename T>
struct void_return_type {
  using type = T;
};

template <>
struct void_return_type<void> {
  using type = int;
};

template <typename T>
using void_return_type_t = typename void_return_type<T>::type;

template <typename T>
struct ptr {
  using value_type = T;
  using element_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  using rebind = ptr<U>;

  ptr(const ptr&) = default;
  ptr& operator=(const ptr&) = default;
  ptr(ptr&&) = default;

  ptr(size_t offset = 0) : offset_(offset) {}

  template <typename ET>
  static auto pointer_to(ET& r) {
    char* raw_ptr = (char *) std::addressof(r);
    size_t offset = raw_ptr - segment_ptr_;
    return ptr<ET>(offset);
  }

  /*
  static element_type* to_address(ptr p) noexcept {
    return local();
  }
  */

  bool operator==(const std::nullptr_t) const {
    return offset_ == 0;
  }

  bool operator!=(const std::nullptr_t) const {
    return !(*this == nullptr);
  }

  operator ptr<void>() const {
    return ptr<void>(offset_);
  }

  operator bool() const {
    return *this != nullptr;
  }

  T* local() const {
    return reinterpret_cast<T*>(segment_ptr_ + offset_);
  }

  void_return_type_t<T>& operator*() const {
    void_return_type_t<T>* ptr = reinterpret_cast<void_return_type_t<T>*>(local());
    return *ptr;
  }

  void_return_type_t<T>& operator[](difference_type idx) const {
    return *(*this + idx);
  }

  auto operator->() const {
    if constexpr(!std::is_void_v<T>) {
      return local();
    }
  }

  ptr<T> operator+(difference_type offset) const {
    return ptr<T>(offset_ + offset*sizeof(T));
  }

  ptr<T> operator-(difference_type offset) const {
    return ptr<T>(offset_ - offset*sizeof(T));
  }

  difference_type operator-(const ptr<T>& ptr) const {
    return (offset_ - ptr.offset_) / sizeof(T);
  }

  ptr<T> operator++(int) {
    ptr<T> ptr = *this;
    offset_ += sizeof(T);
    return ptr;
  }

  ptr<T> operator--(int) {
    ptr<T> ptr = *this;
    offset_ -= sizeof(T);
    return ptr;
  }

  ptr<T>& operator++() {
    offset_ += sizeof(T);
    return *this;
  }

  ptr<T>& operator--() {
    offset_ -= sizeof(T);
    return *this;
  }

  ptr<T>& operator+=(difference_type offset) {
    offset_ += offset*sizeof(T);
    return *this;
  }

  ptr<T>& operator-=(difference_type offset) {
    offset_ -= offset*sizeof(T);
    return *this;
  }

  bool operator==(const ptr<T>& other) const {
    return offset_ = other.ofset_;
  }

  bool operator!=(const ptr<T>& other) const {
    return !(*this == other);
  }

  std::string str() const {
    return "shm{" + std::to_string(offset_) + "}";
  }

  size_t offset_ = 0;
};

} // end shm
