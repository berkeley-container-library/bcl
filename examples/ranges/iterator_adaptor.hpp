#pragma once

template <typename Accessor>
class random_access_iterator_adaptor {
public:

  using accessor_type = Accessor;
  using const_accessor_type = typename Accessor::const_iterator_accessor;
  using nonconst_accessor_type = typename Accessor::nonconst_iterator_accessor;

  using difference_type = typename Accessor::difference_type;
  using value_type = typename Accessor::value_type;
  using iterator = random_access_iterator_adaptor<accessor_type>;
  using const_iterator = random_access_iterator_adaptor<const_accessor_type>;
  using reference = typename Accessor::reference;
  using iterator_category = typename Accessor::iterator_category;

  using nonconst_iterator = random_access_iterator_adaptor<nonconst_accessor_type>;

  static_assert(std::is_same_v<iterator, random_access_iterator_adaptor<Accessor>>);

  random_access_iterator_adaptor() = default;
  ~random_access_iterator_adaptor() = default;
  random_access_iterator_adaptor(const random_access_iterator_adaptor&) = default;
  random_access_iterator_adaptor& operator=(const random_access_iterator_adaptor&) = default;

  template <typename... Args>
  requires(sizeof...(Args) >= 1                                        &&
           !(std::is_same_v<nonconst_iterator, std::decay_t<Args>> ||...)       &&
           !(std::is_same_v<const_iterator, std::decay_t<Args>> ||...) &&
           !(std::is_same_v<nonconst_accessor_type, std::decay_t<Args>> ||...)  &&
           !(std::is_same_v<const_accessor_type, std::decay_t<Args>> ||...))
  random_access_iterator_adaptor(Args&&... args)
    : accessor_(std::forward<Args>(args)...) {}

  random_access_iterator_adaptor(const accessor_type& accessor) : accessor_(accessor) {}
  random_access_iterator_adaptor(const const_accessor_type& accessor)
  requires(!std::is_same_v<accessor_type, const_accessor_type>)
  : accessor_(accessor) {}

  operator const_iterator() const
  requires(!std::is_same_v<iterator, const_iterator>)
  {
    return const_iterator(accessor_);
  }

  bool operator==(const_iterator other) const {
    return accessor_ == other.accessor_;
  }

  bool operator!=(const_iterator other) const {
    return !(*this == other);
  }

  bool operator<(const_iterator other) const {
    return accessor_ < other.accessor_;
  }

  bool operator<=(const_iterator other) const {
    return *this < other || *this == other;
  }

  bool operator>(const_iterator other) const {
    return !(*this <= other);
  }

  bool operator>=(const_iterator other) const {
    return !(*this < other);
  }

  reference operator*() const {
    return *accessor_;
  }

  reference operator[](difference_type offset) const {
    return *(*this + offset);
  }

  iterator& operator+=(difference_type offset) noexcept {
    accessor_ += offset;
    return *this;
  }

  iterator& operator-=(difference_type offset) noexcept {
    accessor_ += -offset;
    return *this;
  }

  iterator operator+(difference_type offset) const {
    iterator other = *this;
    other += offset;
    return other;
  }

  iterator operator-(difference_type offset) const {
    iterator other = *this;
    other += -offset;
    return other;
  }

  difference_type operator-(const_iterator other) const {
    return accessor_ - other.accessor_;
  }

  iterator& operator++() noexcept {
    *this += 1;
    return *this;
  }

  iterator operator++(int) noexcept {
    iterator other = *this;
    ++(*this);
    return other;
  }

  iterator& operator--() noexcept {
    *this += -1;
    return *this;
  }

  iterator operator--(int) noexcept {
    iterator other = *this;
    --(*this);
    return other;
  }

  friend iterator operator+(difference_type n, iterator iter) {
    return iter + n;
  }

private:

  friend const_iterator;
  friend nonconst_iterator;

  accessor_type accessor_;
};

template <std::integral I>
class iota_accessor {
public:
  using value_type = I;
  using difference_type = std::ptrdiff_t;
  using iterator_accessor = iota_accessor;
  using const_iterator_accessor = iota_accessor;
  using reference = I;
  using iterator_category = std::random_access_iterator_tag;

  constexpr iota_accessor() noexcept = default;
  constexpr ~iota_accessor() noexcept = default;
  constexpr iota_accessor(const iota_accessor&) noexcept = default;
  constexpr iota_accessor& operator=(const iota_accessor&) noexcept = default;

  constexpr iota_accessor(I value) noexcept : value_(value) {}

  constexpr iota_accessor& operator+=(difference_type offset) noexcept {
    value_ += offset;
    return *this;
  }

  constexpr difference_type operator-(const const_iterator_accessor& other) const noexcept {
    return value_ - other.value_;
  }

  constexpr bool operator==(const const_iterator_accessor& other) const noexcept {
    return value_ == other.value_;
  }

  constexpr bool operator<(const const_iterator_accessor& other) const noexcept {
    return value_ < other.value_;
  }

  constexpr reference operator*() const noexcept {
    return value_;
  }

private:
  I value_;
};

template <std::integral I = std::size_t>
using iota_iterator = random_access_iterator_adaptor<iota_accessor<I>>;