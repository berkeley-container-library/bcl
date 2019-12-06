//
// Created by Jiakun Yan on 12/3/19.
//

#include <iostream>
#include <thread>
#include <functional>

void foo() {
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
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  std::thread t(handler<Fn, Args...>, std::forward<Fn>(fn),
                std::forward<Args>(args)...);
  t.join();
}

int main() {
  run(&foo);

  return 0;
}