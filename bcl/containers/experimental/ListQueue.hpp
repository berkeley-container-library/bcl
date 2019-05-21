#include <bcl/bcl.hpp>
#include <unistd.h>

namespace BCL {

template <typename T>
struct QueueNode {
  T value_;
  int flag_ = 0;
  BCL::GlobalPtr<QueueNode<T>> next = nullptr;
  bool occupied_ = 0;

  QueueNode(const QueueNode &) = default;
  QueueNode() = default;
  QueueNode(const T& value) : value_(value), occupied_(1) {}
  QueueNode(T&& value) : value_(std::move(value)), occupied_(1) {}
};

template <typename T>
struct QueueIterator {
  BCL::GlobalPtr<QueueNode<T>> loc_ = nullptr;

  bool operator==(const QueueIterator<T>& other) const {
    return other.loc_ == loc_;
  }

  bool operator!=(const QueueIterator<T>& other) const {
    return !(*this == other);
  }

  BCL::GlobalRef<T> operator*() {
    return *pointerto(value_, loc_);
  }

  auto operator++() {
    loc_ = *pointerto(next, loc_);
    return *this;
  }

  auto operator++(int) {
    loc_ = *pointerto(next, loc_);
    return *this;
  }

  QueueIterator(const QueueIterator&) = default;
  QueueIterator(BCL::GlobalPtr<QueueNode<T>> loc) : loc_(loc) {}
  QueueIterator() {}
};

template <typename T>
struct ListQueue {
  BCL::GlobalPtr<QueueNode<T>> head = nullptr;
  BCL::GlobalPtr<QueueNode<T>> tail_buf = nullptr;
  BCL::GlobalPtr<QueueNode<T>> head_buf_ = nullptr;

  BCL::GlobalPtr<QueueNode<T>> head_buf() {
    if (head_buf_ == nullptr) {
      head_buf_ = *pointerto(next, head);
      return head_buf_;
    }
    return head_buf_;
  }

  QueueIterator<T> begin() {
    return QueueIterator<T>(head_buf());
  }

  QueueIterator<T> end() {
    return QueueIterator<T>(nullptr);
  }

  auto back_iter() {
    advance_tail_buf();
    return QueueIterator<T>(tail_buf);
  }

  // TODO: should be a reference.
  T back() {
    advance_tail_buf();
    return *QueueIterator<T>(tail_buf);
  }

  void advance_tail_buf() {
    while (BCL::rget(pointerto(next, tail_buf)) != nullptr) {
      tail_buf = *pointerto(next, tail_buf);
    }
  }

  ListQueue(size_t host) {
    if (BCL::rank() == host) {
      head = BCL::alloc<QueueNode<T>>(1);
      new (head.local()) QueueNode<T>();
    }

    head = BCL::broadcast(head, host);
    tail_buf = head;
  }

  ListQueue(ListQueue&&) = default;

  void lock(BCL::GlobalPtr<QueueNode<T>> node) {
    auto flag = pointerto(flag_, node);
    int old_val = BCL::compare_and_swap(flag, 0, 1);
    size_t iter = 1;
    while (old_val != 0) {
      usleep(1 << iter++);
      old_val = BCL::compare_and_swap(flag, 0, 1);
    }
  }

  void unlock(BCL::GlobalPtr<QueueNode<T>> node) const {
    auto flag = pointerto(flag_, node);
    int old_val = BCL::compare_and_swap(flag, 1, 0);
    if (old_val != 1) {
      throw std::runtime_error("AGH! tried to unlock node " + node.str() + ", but somebody"
            " else did that for me, got value " + std::to_string(old_val));
    }
  }

  // TODO: if we still had RPCs, could buffer remote memory
  void push(const T& value) {
    lock(tail_buf);
    auto cur_next = BCL::rget(pointerto(next, tail_buf));
    if (cur_next == nullptr) {
      auto new_node = BCL::alloc<BCL::QueueNode<T>>(1);
      new (new_node.local()) BCL::QueueNode<T>(value);
      BCL::rput(new_node, pointerto(next, tail_buf));
      BCL::flush();
      unlock(tail_buf);
    } else {
      unlock(tail_buf);
      tail_buf = cur_next;
      push(value);
    }
  }

  void push(T&& value) {
    lock(tail_buf);
    auto cur_next = BCL::rget(pointerto(next, tail_buf));
    if (cur_next == nullptr) {
      auto new_node = BCL::alloc<BCL::QueueNode<T>>(1);
      new (new_node.local()) BCL::QueueNode<T>(std::move(value));
      BCL::rput(new_node, pointerto(next, tail_buf));
      BCL::flush();
      unlock(tail_buf);
    } else {
      unlock(tail_buf);
      tail_buf = cur_next;
      push(std::move(value));
    }
  }

  // Attempt to insert `value` after iterator `iter`
  bool push_after(const T& value, QueueIterator<T> iter) {
    lock(iter.loc_);
    auto cur_next = BCL::rget(pointerto(next, iter.loc_));
    if (cur_next == nullptr) {
      auto new_node = BCL::new_<BCL::QueueNode<T>>(value);
      BCL::rput(new_node, pointerto(next, tail_buf));
      BCL::flush();
      unlock(iter.loc_);
      return true;
    } else {
      unlock(iter.loc_);
      return false;
    }
  }

  bool push_after(T&& value, QueueIterator<T> iter) {
    lock(iter.loc_);
    auto cur_next = BCL::rget(pointerto(next, iter.loc_));
    if (cur_next == nullptr) {
      auto new_node = BCL::new_<BCL::QueueNode<T>>(std::move(value));
      BCL::rput(new_node, pointerto(next, iter.loc_));
      BCL::flush();
      unlock(iter.loc_);
      return true;
    } else {
      unlock(iter.loc_);
      return false;
    }
  }

  template <typename... Args>
  bool emplace_after(QueueIterator<T> iter, Args&&... args) {
    lock(iter.loc_);
    auto cur_next = BCL::rget(pointerto(next, iter.loc_));
    if (cur_next == nullptr) {
      auto new_node = BCL::new_<BCL::QueueNode<T>>();
      auto nnlocal = new_node.local();
      nnlocal->occupied_ = 1;
      new (&nnlocal->value_) T(std::forward<Args>(args)...);
      BCL::rput(new_node, pointerto(next, iter.loc_));
      BCL::flush();
      unlock(iter.loc_);
      return true;
    } else {
      unlock(iter.loc_);
      return false;
    }
  }

  void print() {
    size_t i = 0;
    for (auto node = head; node != nullptr; node = node->next) {
      if (node->occupied_) {
        printf("Value %lu: %d\n", i++, node->value_);
      }
    }
  }
};

}
