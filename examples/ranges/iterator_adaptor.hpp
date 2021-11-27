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
  using nonconst_iterator_accessor = iota_accessor;
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

template<std::integral I = std::size_t>
class iota_view {
public:
  using iterator = iota_iterator<I>;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using size_type = std::size_t;
  using reference = typename iterator::reference;
  using pointer = iterator;

  iota_view(I end = std::numeric_limits<I>::max()) : begin_(iterator(0)), end_(iterator(end)) {}

  iota_view(I begin, I end) : begin_(iterator(begin)), end_(iterator(end)) {}

  iterator begin() const noexcept {
    return begin_;
  }

  iterator end() const noexcept {
    return end_;
  }

  size_type size() const noexcept {
    return end() - begin();
  }

private:
  iterator begin_;
  iterator end_;
};

template <std::random_access_iterator... Its>
class zip_accessor {
public:
  using value_type = std::tuple<std::iter_value_t<Its>...>;
  using difference_type = std::ptrdiff_t;
  using iterator_accessor = zip_accessor;
  using const_iterator_accessor = zip_accessor;
  using nonconst_iterator_accessor = zip_accessor;
  using reference = std::tuple<std::iter_reference_t<Its>...>;
  using iterator_category = std::random_access_iterator_tag;

  constexpr zip_accessor() noexcept = default;
  constexpr ~zip_accessor() noexcept = default;
  constexpr zip_accessor(const zip_accessor&) noexcept = default;
  constexpr zip_accessor& operator=(const zip_accessor&) noexcept = default;

  constexpr zip_accessor(Its... iterators, std::size_t offset = 0) noexcept : iterators_(iterators...), offset_(offset) {}

  constexpr zip_accessor& operator+=(difference_type offset) noexcept {
    offset_ += offset;
    return *this;
  }

  constexpr difference_type operator-(const const_iterator_accessor& other) const noexcept {
    return offset_ - other.offset_;
  }

  constexpr bool operator==(const const_iterator_accessor& other) const noexcept {
    return offset_ == other.offset_;
  }

  constexpr bool operator<(const const_iterator_accessor& other) const noexcept {
    return offset_ < other.offset_;
  }

  constexpr reference operator*() const noexcept {
    return get_reference_impl_(std::make_index_sequence<sizeof...(Its)>());
  }

private:
  template <std::size_t... Is>
  reference get_reference_impl_(std::index_sequence<Is...> idx_seq) const noexcept {
    return reference(std::get<Is>(iterators_)[offset_]...);
  }

private:
  std::size_t offset_;
  std::tuple<Its...> iterators_;
};

template <std::random_access_iterator... Its>
using zip_iterator = random_access_iterator_adaptor<zip_accessor<Its...>>;

template <std::random_access_iterator... Its>
zip_iterator<Its...> make_zip_iterator(std::size_t offset, Its... iterators) {
  return zip_iterator<Its...>(iterators..., offset);
}

template <std::ranges::random_access_range... Rs>
class zip_view {
public:
  using iterator = zip_iterator<std::ranges::iterator_t<Rs>...>;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using size_type = std::size_t;
  using reference = typename iterator::reference;
  using pointer = iterator;

  zip_view(Rs&&... ranges)
  : begin_(make_zip_iterator(0, std::ranges::begin(ranges)...)) { size_ = smallest_range(ranges...); }

  zip_view(Rs&... ranges)
  : begin_(make_zip_iterator(0, std::ranges::begin(ranges)...)) { size_ = smallest_range(ranges...); }

  zip_view(const Rs&... ranges)
  : begin_(make_zip_iterator(0, std::ranges::begin(ranges)...)) { size_ = smallest_range(ranges...); }

  iterator begin() const noexcept {
    return begin_;
  }

  iterator end() const noexcept {
    return begin() + size_;
  }

  size_type size() const noexcept {
    return size_;
  }

private:
  template <typename R_>
  std::size_t smallest_range_impl_(R_&& range) { return std::ranges::size(std::forward<R_>(range)); }
  template <typename R_, typename... Rs_>
  std::size_t smallest_range_impl_(R_&& range, Rs_&&... ranges) {
    return std::min(std::ranges::size(std::forward<R_>(range)), smallest_range_impl_(std::forward<Rs_>(ranges)...));
  }

  template <typename... Rs_>
  std::size_t smallest_range(Rs_&&... ranges) {
    return smallest_range_impl_(std::forward<Rs_>(ranges)...);
  }

  iterator begin_;
  std::size_t size_;
};

class transform_no_inverse {};

template <typename Reference, typename Fn, typename FnInverse>
class transform_reference {
public:

  using value_type = decltype(std::declval<Fn>()(std::declval<Reference>()));

  template <typename Reference_>
  transform_reference(Reference_&& reference, Fn fn, FnInverse fn_inverse)
    : reference_(std::forward<Reference_>(reference)),
      fn_(fn),
      fn_inverse_(fn_inverse) {}

  operator value_type() const {
    return fn_(reference_);
  }

  transform_reference& operator=(const value_type& value)
  requires(!std::is_same_v<FnInverse, transform_no_inverse>)
  {
    reference_ = fn_inverse_(value);
    return *this;
  }

private:
  Reference reference_;
  Fn fn_;
  FnInverse fn_inverse_;
};

template <std::random_access_iterator Iter, typename Fn, typename FnInverse = transform_no_inverse>
class transform_accessor {
public:
  using value_type = std::remove_cvref_t<decltype(std::declval<Fn>()(*std::declval<Iter>()))>;
  using difference_type = std::ptrdiff_t;
  using iterator_accessor = transform_accessor;
  using const_iterator_accessor = transform_accessor;
  using nonconst_iterator_accessor = transform_accessor;
  using iterator_category = std::random_access_iterator_tag;

  using reference = transform_reference<std::iter_reference_t<Iter>, Fn, FnInverse>;

  constexpr transform_accessor() noexcept = default;
  constexpr ~transform_accessor() noexcept = default;
  constexpr transform_accessor(const transform_accessor&) noexcept = default;
  constexpr transform_accessor& operator=(const transform_accessor&) noexcept = default;

  constexpr transform_accessor(Iter iterator, Fn fn, FnInverse fn_inverse) noexcept
    : iterator_(iterator), fn_(fn), fn_inverse_(fn_inverse) {}

  constexpr transform_accessor& operator+=(difference_type offset) noexcept {
    iterator_ += offset;
    return *this;
  }

  constexpr difference_type operator-(const const_iterator_accessor& other) const noexcept {
    return iterator_ - other.iterator_;
  }

  constexpr bool operator==(const const_iterator_accessor& other) const noexcept {
    return iterator_ == other.iterator_;
  }

  constexpr bool operator<(const const_iterator_accessor& other) const noexcept {
    return iterator_ < other.iterator_;
  }

  constexpr reference operator*() const noexcept {
    return reference(*iterator_, fn_, fn_inverse_);
  }

private:
  Iter iterator_;
  Fn fn_;
  FnInverse fn_inverse_;
};


template <std::random_access_iterator Iter, typename Fn, typename FnInverse = transform_no_inverse>
using transform_iterator = random_access_iterator_adaptor<transform_accessor<Iter, Fn, FnInverse>>;

template <std::random_access_iterator Iter, typename Fn, typename FnInverse = transform_no_inverse>
transform_iterator<Iter, Fn, FnInverse> make_transform_iterator(Iter iterator, Fn fn, FnInverse fn_inverse = FnInverse()) {
  return transform_iterator<Iter, Fn, FnInverse>(iterator, fn, fn_inverse);
}

template <std::ranges::random_access_range R, typename Fn, typename FnInverse = transform_no_inverse>
class transform_view {
public:
  using iterator = transform_iterator<std::ranges::iterator_t<R>, Fn, FnInverse>;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using size_type = std::size_t;
  using reference = typename iterator::reference;
  using pointer = iterator;

  transform_view(R&& range, Fn fn, FnInverse fn_inverse = FnInverse())
    : begin_(make_transform_iterator(std::ranges::begin(range), fn, fn_inverse)),
      end_(make_transform_iterator(std::ranges::end(range), fn, fn_inverse)) {}

  transform_view(R& range, Fn fn, FnInverse fn_inverse = FnInverse())
    : begin_(make_transform_iterator(std::ranges::begin(range), fn, fn_inverse)),
      end_(make_transform_iterator(std::ranges::end(range), fn, fn_inverse)) {}

  transform_view(const R& range, Fn fn, FnInverse fn_inverse = FnInverse())
    : begin_(make_transform_iterator(std::ranges::begin(range), fn, fn_inverse)),
      end_(make_transform_iterator(std::ranges::end(range), fn, fn_inverse)) {}

  iterator begin() const noexcept {
    return begin_;
  }

  iterator end() const noexcept {
    return end_;
  }

  size_type size() const noexcept {
    return end_ - begin_;
  }

private:
  iterator begin_;
  iterator end_;
};
