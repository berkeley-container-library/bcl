#pragma once

#include <bcl/bcl.hpp>
#include <execution>
#include <bcl/core/teams.hpp>
#include <ranges>
#include "remote_span.hpp"
#include "iterator_adaptor.hpp"

template <typename T>
class distributed_span_accessor {
public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using iterator_accessor = distributed_span_accessor;
  using const_iterator_accessor = distributed_span_accessor<std::add_const_t<T>>;
  using reference = BCL::GlobalRef<T>;
  using iterator_category = std::random_access_iterator_tag;

  using nonconst_iterator_accessor = distributed_span_accessor<std::remove_const_t<T>>;

  using local_span_type = remote_span<std::remove_const_t<T>>;

  constexpr distributed_span_accessor() noexcept = default;
  constexpr ~distributed_span_accessor() noexcept = default;
  constexpr distributed_span_accessor(const distributed_span_accessor&) noexcept = default;
  constexpr distributed_span_accessor& operator=(const distributed_span_accessor&) noexcept = default;

  constexpr distributed_span_accessor(std::span<local_span_type> spans, std::size_t index = 0) noexcept : spans_(spans), index_(index) {}

  constexpr operator const_iterator_accessor() const noexcept
  requires(!std::is_same_v<iterator_accessor, const_iterator_accessor>)
  {
    return const_iterator_accessor(spans_, index_);
  }

  constexpr distributed_span_accessor& operator+=(difference_type offset) noexcept {
    index_ += offset;
    return *this;
  }

  constexpr difference_type operator-(const const_iterator_accessor& other) const noexcept {
    return difference_type(index_) - other.index_; 
  }

  constexpr bool operator==(const const_iterator_accessor& other) const noexcept {
    return index_ == other.index_;
  }

  constexpr bool operator<(const const_iterator_accessor& other) const noexcept {
    return index_ < other.index_;
  }

  constexpr reference operator*() const noexcept {
    return spans_[index_ / local_size()][index_ % local_size()];
  }

  constexpr std::size_t local_size() const noexcept {
    return spans_.front().size();
  }


private:
  friend const_iterator_accessor;
  friend distributed_span_accessor<std::remove_const_t<T>>;

  std::size_t index_;
  std::span<local_span_type> spans_;
};

template <typename T>
using distributed_span_iterator = random_access_iterator_adaptor<distributed_span_accessor<T>>;

template <typename T>
class distributed_span {
public:
  using local_span_type = remote_span<T>;

  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using pointer = typename local_span_type::pointer;
  using const_pointer = typename local_span_type::const_pointer;

  using reference = typename local_span_type::reference;
  using const_reference = typename local_span_type::const_reference;

  using joined_view_type = std::ranges::join_view<std::ranges::ref_view<std::vector<local_span_type>>>;

  // using iterator = std::ranges::iterator_t<joined_view_type>;
  using iterator = distributed_span_iterator<T>;

  constexpr distributed_span() noexcept = default;
  constexpr distributed_span(const distributed_span&) noexcept = default;
  constexpr distributed_span& operator=(const distributed_span&) noexcept = default;

  template <std::ranges::input_range R>
  requires(std::is_same_v<std::ranges::range_value_t<R>, local_span_type>)
  constexpr distributed_span(R&& spans)
    : spans_(std::ranges::begin(spans), std::ranges::end(spans)), joined_view_(spans_),
      local_joined_view_(set_local_views(spans_))
    {
      for (size_t i = 0; i < spans_.size(); i++) {
        size_ += spans_[i].size();
        if (spans_[i].size() != local_size() && i != spans_.size() - 1) {
          throw std::runtime_error("Error! distributed_span given non-uniform sizes");
        }
      }
    }

  constexpr size_type size() const noexcept {
    return size_;
  }

  constexpr size_type size_bytes() const noexcept {
    return size()*sizeof(element_type);
  }

  constexpr reference operator[](size_type idx) const {
    return spans_[idx / local_size()][idx % local_size()];
  }

  constexpr std::size_t local_size(size_type rank = 0) const noexcept {
    return spans_[rank].size();
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size() == 0;
  }

  constexpr distributed_span<element_type>
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const {
    std::vector<local_span_type> new_spans;

    size_type local_id = Offset / local_size();
    size_type local_offset = Offset % local_size();
    size_type local_count = std::min(local_size() - local_offset, Count);

    new_spans.push_back(spans_[local_id].subspan(local_offset, local_count));

    local_id++;

    for (; local_id*local_size() < Offset + Count && local_id < spans_.size(); local_id++) {
      size_type local_count = std::min(local_size(), (Offset + Count) - local_id*local_size());
      new_spans.push_back(spans_[local_id].subspan(0, local_count));
    }

    return distributed_span<element_type>(new_spans);
  }

  constexpr distributed_span<element_type> first(size_type Count) const {
    return subspan(0, Count);
  }

  constexpr distributed_span<element_type> last(size_type Count) const {
    return subspan(size() - Count, Count);
  }

  iterator begin() {
    // using accessor_type = typename iterator::accessor_type;
    // return iterator(accessor_type(spans_, 0));
    return iterator(spans_, 0);
  }

  iterator end() {
    // using accessor_type = typename iterator::accessor_type;
    // return iterator(accessor_type(spans_, 0));
    return iterator(spans_, size());
  }

/*
  iterator local_begin() const {
    return local_joined_view_.begin();
  }

  iterator local_end() const {
    return local_joined_view_.end();
  }
  */

  constexpr reference front() const {
    return spans_.front().front();
  }

  constexpr reference back() const {
    return spans_.back().back();
  }


private:
  std::vector<local_span_type>& set_local_views(const std::vector<local_span_type>& spans) {
    my_spans_.resize(0);
    for (auto&& span : spans_) {
      if (span.data().rank == BCL::rank()) {
        my_spans_.push_back(span);
      }
    }
    return my_spans_;
  }

private:
  std::size_t size_ = 0;
  std::vector<local_span_type> spans_;
  joined_view_type joined_view_;
  std::vector<local_span_type> my_spans_;
  joined_view_type local_joined_view_;
};

