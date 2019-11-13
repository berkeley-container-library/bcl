//
// Created by Jiakun Yan on 11/12/19.
//
#ifndef BCL_ARH_AVERAGETIMER_HPP
#define BCL_ARH_AVERAGETIMER_HPP
namespace ARH {

  extern size_t my_worker_local();

  struct AverageTimer {
  private:
    alignas(alignof_cacheline) unsigned long step = 0;
    alignas(alignof_cacheline) tick_t _start = 0;
    alignas(alignof_cacheline) double _ticks = 0;
  public:
    void start() {
      if (my_worker_local() == 0) {
        _start = ticks_now();
      } else {
        ticks_now();
      }
    }

    void end_and_update() {
      tick_t _end = ticks_now();
      if (my_worker_local() == 0) {
        update_average(_ticks, _end - _start, ++step);
      }
    }

    void tick_and_update(tick_t _start_) {
      tick_t _end = ticks_now();
      if (my_worker_local() == 0) {
        update_average(_ticks, _end - _start_, ++step);
      }
    }

    [[nodiscard]] double to_ns() const {
      return ticks_to_ns(_ticks);
    }

    [[nodiscard]] double to_us() const {
      return ticks_to_ns(_ticks) / 1e3;
    }

    [[nodiscard]] double to_s() const {
      return ticks_to_ns(_ticks) / 1e9;
    }

    void print_us(std::string &&name = "") const {
      if (my_worker_local() == 0) {
        printf("Duration %s: %.3lf us\n", name.c_str(), to_us());
      }
    }
  };

  struct SimpleTimer {
  private:
    unsigned long step = 0;
    tick_t _start = 0;
    double _ticks = 0;
  public:
    void start() {
      _start = ticks_now();
    }

    void end_and_update() {
      tick_t _end = ticks_now();
      update_average(_ticks, _end - _start, ++step);
    }

    void tick_and_update(tick_t _start_) {
      tick_t _end = ticks_now();
      update_average(_ticks, _end - _start_, ++step);
    }

    [[nodiscard]] double to_ns() const {
      return ticks_to_ns(_ticks);
    }

    [[nodiscard]] double to_us() const {
      return ticks_to_ns(_ticks) / 1e3;
    }

    [[nodiscard]] double to_s() const {
      return ticks_to_ns(_ticks) / 1e9;
    }

    void print_us(std::string &&name = "") const {
      printf("Duration %s: %.3lf us\n", name.c_str(), to_us());
    }
  };
}
#endif //BCL_ARH_AVERAGETIMER_HPP
