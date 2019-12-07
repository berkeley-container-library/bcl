//
// Created by Jiakun Yan on 12/3/19.
//

#include <iostream>
#include <thread>
#include <functional>

void foo(int val) {
  std::printf("hello world\n");
}


template <typename Fn, typename... Args>
void handler(Fn&& fn, Args &&... args) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  std::invoke(std::forward<Fn>(fn),
              std::forward<Args>(args)...);
}

template <typename Fn, typename... Args>
void run(Fn&& fn, Args &&... args) {
  using fn_t = decltype(+std::declval<std::remove_reference_t<Fn>>());
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  std::thread t(handler<fn_t, std::remove_reference_t<Args>...>, std::forward<fn_t>(+fn),
                std::forward<Args>(args)...);
  t.join();
}

int main() {
  int val = 1;
  run(foo, val);

  return 0;
}