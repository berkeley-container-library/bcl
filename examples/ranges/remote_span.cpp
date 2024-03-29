#include <bcl/bcl.hpp>
#include <iostream>

#ifndef CPP20
#include <limits>
namespace std {
inline constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();
}
#else
#include <span>
#endif

template <typename T,
          std::size_t Extent = std::dynamic_extent>
struct remote_span_storage_impl_ {
  BCL::GlobalPtr<T> ptr_;
  remote_span_storage_impl_(BCL::GlobalPtr<T> ptr, std::size_t size) : ptr_(ptr) {}
  constexpr std::size_t size() const noexcept {
    return Extent;
  }
};

template <typename T>
struct remote_span_storage_impl_<T, std::dynamic_extent> {
  BCL::GlobalPtr<T> ptr_;
  std::size_t size_;

  remote_span_storage_impl_(BCL::GlobalPtr<T> ptr, std::size_t size) : ptr_(ptr), size_(size) {}

  constexpr std::size_t size() const noexcept {
    return size_;
  }
};

template <typename T,
          std::size_t Extent = std::dynamic_extent>
class remote_span {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using pointer = BCL::GlobalPtr<T>;
  using const_pointer = BCL::GlobalPtr<const T>;

  using reference = BCL::GlobalRef<T>;
  using const_reference = BCL::GlobalRef<const T>;

  using iterator = pointer;

  constexpr remote_span(BCL::GlobalPtr<T> first, size_type count) : storage_(first, count) {}

  constexpr remote_span(BCL::GlobalPtr<T> first, BCL::GlobalPtr<T> last)
    : storage_(first, last - first) {}

  constexpr size_type size() const noexcept {
    return storage_.size();
  }

  constexpr size_type size_bytes() const noexcept {
    return size()*sizeof(element_type);
  }

  constexpr pointer data() const noexcept {
    return storage_.ptr_;
  }

  constexpr iterator begin() const noexcept {
    return storage_.ptr_;
  }

  constexpr iterator end() const noexcept {
    return data() + size();
  }

  constexpr reference front() const {
    return data();
  }

  constexpr reference back() const {
    return data() + (size() - 1);
  }

  constexpr reference operator[](size_type idx) const {
    return data()[idx];
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size() == 0;
  }

  template <std::size_t Count>
  constexpr remote_span<element_type, Count> first() const {
    return remote_span<element_type, Count>(data(), Count);
  }
      
  constexpr remote_span<element_type, std::dynamic_extent> first(size_type Count) const {
    return remote_span<element_type, std::dynamic_extent>(data(), Count);
  }

  template <std::size_t Count>
  constexpr remote_span<element_type, Count> last() const {
    return remote_span<element_type, Count>(data() + size() - Count, Count);
  }
      
  constexpr remote_span<element_type, std::dynamic_extent> last( size_type Count ) const {
    return remote_span<element_type, std::dynamic_extent>(data() + size() - Count, Count);
  }

  template<std::size_t Offset,
           std::size_t Count = std::dynamic_extent>
  constexpr auto subspan() const {
    if constexpr(Count != std::dynamic_extent) {
      return remote_span<element_type, Count>(data() + Offset, Count);
    } else if constexpr(Extent != std::dynamic_extent) {
      return remote_span<element_type, Extent - Offset>(data() + Offset, Count);
    } else {
      return remote_span<element_type, std::dynamic_extent>(data() + Offset, size() - Offset);
    }
  }

  constexpr remote_span<element_type, std::dynamic_extent>
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const {
    if (Count == std::dynamic_extent) {
      return remote_span<element_type, std::dynamic_extent>(data() + Offset, size() - Offset);
    } else {
      return remote_span<element_type, std::dynamic_extent>(data() + Offset, Count);
    }
  }

private:

  remote_span_storage_impl_<T, Extent> storage_;
};

int main(int argc, char** argv) {
  BCL::init();

  constexpr std::size_t size_per_proc = 10;

  BCL::GlobalPtr<int> data = nullptr;

  srand48(BCL::rank());

  if (BCL::rank() == 0) {
    data = BCL::alloc<int>(size_per_proc*BCL::nprocs());
  }

  data = BCL::broadcast(data, 0);

  remote_span<int> span(data, size_per_proc*BCL::nprocs());

  printf("Span is size %lu\n", span.size());

  auto my_span = span.subspan(size_per_proc*BCL::rank(), size_per_proc);

  printf("My Span is size %lu\n", span.size());

  std::transform(my_span.begin(), my_span.end(), my_span.begin(),
                 [](auto ref) { return lrand48() % 100; });

  BCL::barrier();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < span.size(); i++) {
      int v = span[i];
      std::cout << "(" << v << ")";
    }
    std::cout << std::endl;
  }

  if (BCL::rank() == 0) {
    std::sort(span.begin(), span.end());
  }

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < span.size(); i++) {
      int v = span[i];
      std::cout << "(" << v << ")";
    }
    std::cout << std::endl;
  }

  BCL::finalize();
  return 0;
}