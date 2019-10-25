//
// Created by Jiakun Yan on 10/23/19.
//
#include <gasnetex.h>
#include <time.h>
#include <cstdio>
#include <chrono>
#include <pthread.h>

timespec tick_now0() {
  timespec tick;
  clock_gettime(CLOCK_REALTIME, &tick);
  return tick;
}

timespec tick_now1() {
  timespec tick;
  clock_gettime(CLOCK_MONOTONIC, &tick);
  return tick;
}

timespec tick_now2() {
  timespec tick;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tick);
  return tick;
}

timespec tick_now3() {
  timespec tick;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tick);
  return tick;
}

// nsec
long tick_diff(timespec start, timespec end)
{
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp.tv_nsec;
}

long asm_ticks_now() {
  uint64_t _ret;
  uint32_t _lo, _hi;
  __asm__ __volatile__("rdtsc"
  : "=a" (_lo), "=d" (_hi)
  /* no inputs */);
  _ret = ((uint64_t)_lo) | (((uint64_t)_hi)<<32);
  return _ret;
}

void *DoWork(void *arg) {
  size_t num_ops = 10000;

  auto chrono_begin = std::chrono::high_resolution_clock::now();
  timespec time_begin0 = tick_now0();
  timespec time_begin1 = tick_now1();
  timespec time_begin2 = tick_now2();
  timespec time_begin3 = tick_now3();
  long gex_begin = gasneti_ticks_now(); // cputime
  long asm_begin = asm_ticks_now();

  for (size_t i = 0; i < num_ops; i++) {
//    std::chrono::high_resolution_clock::now(); // 373ns
//    tick_now1(); // 370ns
    gasneti_ticks_now(); // 12ns
//    usleep(1);
  }

  auto chrono_end = std::chrono::high_resolution_clock::now();
  timespec time_time_end0 = tick_now0();
  timespec time_end1 = tick_now1();
  timespec time_end2 = tick_now2();
  timespec time_end3 = tick_now3();
  long gex_end = gasneti_ticks_now();
  long asm_end = asm_ticks_now();

  double chrono_duration = std::chrono::duration<double>(chrono_end - chrono_begin).count();
  long time_duration0 = tick_diff(time_begin0, time_time_end0);
  long time_duration1 = tick_diff(time_begin1, time_end1);
  long time_duration2 = tick_diff(time_begin2, time_end2);
  long time_duration3 = tick_diff(time_begin3, time_end3);
  long gex_duration = gasneti_ticks_to_ns(gex_end - gex_begin);
  long asm_duration = (long) ((double) (asm_end - asm_begin)  / 2.3);
  if (*(int *)arg == 0) {
    std::printf("chrono:                    %lf s total %lfus / op s\n", chrono_duration, 1e6*chrono_duration / num_ops);
    std::printf("clock_getclock(REALTIME):  %lf s total %lfus / op s\n", (double)time_duration0 / 1e9, (double)time_duration0 / 1e3 / num_ops);
    std::printf("clock_getclock(MONOTONIC): %lf s total %lfus / op s\n", (double)time_duration1 / 1e9, (double)time_duration1 / 1e3 / num_ops);
    std::printf("clock_getclock(PROCESS):   %lf s total %lfus / op s\n", (double)time_duration2 / 1e9, (double)time_duration2 / 1e3 / num_ops);
    std::printf("clock_getclock(THREAD):    %lf s total %lfus / op s\n", (double)time_duration3 / 1e9, (double)time_duration3 / 1e3 / num_ops);
    std::printf("gex:                       %lf s total %lfus / op s\n", gex_duration / 1e9, (double)gex_duration / 1e3 / num_ops);
    std::printf("asm:                       %lf s total %lfus / op s\n", asm_duration / 1e9, (double)asm_duration / 1e3 / num_ops);
  }

  return NULL;
}

int main() {

  int numberOfThreads = 15;

  pthread_t threads[numberOfThreads];
  int id[numberOfThreads];

  pthread_attr_t attr;
//  cpu_set_t cpus;
  pthread_attr_init(&attr);

  for (int i = 0; i < numberOfThreads; i++) {
//    CPU_ZERO(&cpus);
//    CPU_SET(i, &cpus);
//    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
    id[i] = i;
    pthread_create(&threads[i], &attr, DoWork, &id[i]);
  }

  for (int i = 0; i < numberOfThreads; i++) {
    pthread_join(threads[i], NULL);
  }

  return 0;
}