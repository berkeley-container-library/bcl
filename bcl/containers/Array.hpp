#pragma once

#include <type_traits>

#include <bcl/bcl.hpp>
#include <bcl/containers/Container.hpp>

namespace BCL {

  template <typename T, typename... Args>
  struct first_type {
    using type = T;
  };

  template <typename T, typename... Args>
  using first_type_t = typename first_type<T, Args...>::type;

  template <typename T, typename TSerialize>
  auto container_ptr_rput(const T &src, BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst)
    -> first_type_t<void, BCL::enable_if_t<identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;

    BCL::GlobalPtr <T> casted_dst = reinterpret_pointer_cast <T> (dst);
    BCL::rput(src, casted_dst);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rput(const T &src, BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst)
    -> first_type_t<void, BCL::enable_if_t<!identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;

    CT container = BCL::rget(dst);
    container.set(src);
    BCL::rput(container, dst);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rget(BCL::GlobalPtr <BCL::Container <T, TSerialize>> src)
    -> first_type_t<T, BCL::enable_if_t<identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    BCL::GlobalPtr <T> casted_src = reinterpret_pointer_cast <T> (src);
    return BCL::rget(casted_src);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rget(BCL::GlobalPtr <BCL::Container <T, TSerialize>> src)
    -> first_type_t<T, BCL::enable_if_t<!identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    return BCL::rget(src).get();
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rput(const T *src,
    BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst, const size_t size)
    -> first_type_t<void, BCL::enable_if_t<identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    BCL::GlobalPtr <T> casted_dst = reinterpret_pointer_cast <T> (dst);
    BCL::rput(src, casted_dst, size);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rput(const T *src,
    BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst, const size_t size)
    -> first_type_t<void, BCL::enable_if_t<!identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    std::vector <CT> containers(size);
    for (int i = 0; i < size; i++) {
      containers.set(src[i]);
    }
    BCL::rput(containers.data(), dst, size);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rget(BCL::GlobalPtr <BCL::Container <T, TSerialize>> src,
    T *dst, const size_t size)
    -> first_type_t<void, BCL::enable_if_t<identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    BCL::GlobalPtr <T> casted_src = reinterpret_pointer_cast <T> (src);
    BCL::rget(casted_src, dst, size);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_rget(BCL::GlobalPtr <BCL::Container <T, TSerialize>> src,
    T *dst, const size_t size)
    -> first_type_t<void, BCL::enable_if_t<!identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    std::vector <CT> containers(size);
    BCL::rget(src, containers.data(), size);
    for (int i = 0; i < containers.size(); i++) {
      dst[i] = containers[i].get();
    }
  }

  template <typename T, typename TSerialize>
  auto container_ptr_free(BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst)
    -> first_type_t<void, BCL::enable_if_t<identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
  }

  template <typename T, typename TSerialize>
  auto container_ptr_free(BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst)
    -> first_type_t<void, BCL::enable_if_t<!identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    CT container = BCL::rget(dst);
    container.free();
    BCL::rput(container, dst);
  }

  template <typename T, typename TSerialize>
  auto container_ptr_free(BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst,
    const size_t size)
    -> first_type_t<void, BCL::enable_if_t<identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
  }

  template <typename T, typename TSerialize>
  auto container_ptr_free(BCL::GlobalPtr <BCL::Container <T, TSerialize>> dst,
    const size_t size)
    -> first_type_t<void, BCL::enable_if_t<!identity_serializer<T, TSerialize>::value>>
  {
    using CT = typename BCL::Container <T, TSerialize>;
    std::vector <CT> containers;
    BCL::rget(dst, containers.data(), size);
    for (int i = 0; i < containers.size(); i++) {
      containers[i].free();
    }
    BCL::rput(containers.data(), dst, size);
  }

  template <typename T,
            typename TSerialize = BCL::serialize <T>>
  class ArrayRef;

  template <typename T,
            typename TSerialize = BCL::serialize <T>>
  class Array {
  public:
    size_t my_size = 0;
    uint64_t my_host = 0;

    using CT = BCL::Container <T, TSerialize>;

    BCL::GlobalPtr <CT> data = nullptr;

    size_t size() {
      return my_size;
    }

    uint64_t host() {
      return my_host;
    }

    ArrayRef <T, TSerialize> operator[](size_t idx) {
      return ArrayRef <T, TSerialize> (idx, this);
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    Array(Array&&) = default;
    Array& operator=(Array&&) = default;

    ~Array() {
      if (BCL::rank() == host() && data != nullptr && !BCL::bcl_finalized) {
        for (int i = 0; i < size(); i++) {
          container_ptr_free(data + i);
        }
        dealloc(data);
      }
    }

    Array() = default;

    Array(size_t size) : my_host(BCL::rank()), my_size(size) {
      data = BCL::alloc<CT>(this->size());
      if (data == nullptr) {
          throw std::runtime_error("BCL: Array does not have enough memory");
      }
      CT* ldata = data.local();
      for (int i = 0; i < this->size(); i++) {
        new(&ldata[i]) CT ();
      }
    }

    Array(uint64_t host, size_t size) : my_host(host), my_size(size) {
      if (BCL::rank() == this->host()) {
        data = BCL::alloc <CT> (this->size());
        if (data == nullptr) {
          throw std::runtime_error("BCL: Array does not have enough memory");
        }
        CT* ldata = data.local();
        for (int i = 0; i < this->size(); i++) {
          new(&ldata[i]) CT ();
        }
      }
      data = BCL::broadcast(data, this->host());
    }

    T get(size_t idx) {
      if (idx >= size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      return container_ptr_rget(data + idx);
    }

    void get(size_t idx, std::vector <T> &vals, const size_t size) {
      if (idx + size > this->size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      vals.resize(size);
      container_ptr_rget(data + idx, vals.data(), size);
    }

    void get(size_t idx, T *vals, const size_t size) {
      if (idx + size > this->size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      container_ptr_rget(data + idx, vals, size);
    }

    void put(size_t idx, const T &val) {
      if (idx >= size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      container_ptr_free(data + idx);
      container_ptr_rput(val, data + idx);
    }

    void free(size_t idx) {
      if (idx >= size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      container_ptr_free(data + idx);
    }

    void put(size_t idx, const std::vector <T> &vals) {
      if (idx + vals.size() > size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      for (int i = idx; i < idx + vals.size(); i++) {
        container_ptr_free(data + i);
      }
      container_ptr_rput(vals.data(), data + idx, vals.size());
    }

    void put(size_t idx, const T* src, size_t nelem) {
      if (idx + nelem > size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      for (int i = idx; i < idx + nelem; i++) {
        container_ptr_free(data + i);
      }
      container_ptr_rput(src, data + idx, nelem);
    }

    // An array put with no free.
    // Only different types that serialize to serial_ptr <U>.
    void put_nofree(size_t idx, const T &val) {
      if (idx >= size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      container_ptr_rput(val, data + idx);
    }

    void put_nofree(size_t idx, const std::vector <T> &vals) {
      if (idx + vals.size() > size()) {
        throw std::runtime_error("Array: out of bounds access");
      }
      container_ptr_rput(vals.data(), data + idx, vals.size());
    }
  };

  template <typename T,
            typename TSerialize>
  class ArrayRef {
  public:
    Array <T, TSerialize> *array;
    size_t idx;

    ArrayRef(size_t idx, Array <T, TSerialize> *array) : idx(idx), array(array) {}

    operator T() {
      return array->get(idx);
    }

    T operator*() {
      return array->get(idx);
    }

    T get() {
      return array->get(idx);
    }

    void operator=(const T &val) {
      array->put(idx, val);
    }

    void put(const T &val) {
      array->put(idx, val);
    }

    void free() {
      array->free(idx);
    }

    operator GlobalPtr <T> () {
      return BCL::decay_container(array->data + idx);
    }

    ArrayRef(const ArrayRef &ref) = delete;

    ArrayRef(ArrayRef &&ref) {
      array = ref.array;
      idx = ref.idx;

      ref.array = nullptr;
      ref.idx = 0;
    }
  };
}
