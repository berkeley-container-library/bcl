#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <gasnetex.h>

long ticks_now() {
  timespec tick;
  clock_gettime(CLOCK_MONOTONIC, &tick);
  return tick.tv_nsec / 1e3 + tick.tv_sec * 1e6;
}

long asm_ticks_now() {
  using uint32_t = int;
  using uint64_t = long;
  uint64_t _ret;
  uint32_t _lo, _hi;
  __asm__ __volatile__("rdtsc"
  : "=a" (_lo), "=d" (_hi)
  /* no inputs */);
  _ret = ((uint64_t)_lo) | (((uint64_t)_hi)<<32);
  return _ret;
}

inline void update_average(double &average, long val, long num) {
  average += (val - average) / num;
}

long compute_by_work(double workload) {
  long workload_unit = 10;
  long a = 1, b = 1, c = 0;
  for (long i = 0; i < workload * workload_unit; ++i) {
    c = a + b;
    a = b;
    b = c;
  }
  return b;
}

void* DoWork(void* args) {
//  int records[4];
  int steps = 10000;
  int result = 0;
  long iter = 0;
  double average = 0;

  long begin = ticks_now();
  for (int i = 0; i < steps; ++i) {
    long gex_begin = gasneti_ticks_now();
    result += compute_by_work(steps);
    long gex_end = gasneti_ticks_now();
    update_average(average, gex_end-gex_begin, ++iter);
  }
  long end = ticks_now();

  long duration = end - begin;
  long gex_duration = gasneti_ticks_to_ns(average);
  if (*(int *)args == 0) {
    printf("duration: %.3lf s; ave: %.2lf us; gex_ave: %.3lf us; result: %d\n", (double) duration / 1e6, (double) duration / steps, gex_duration / 1e3, result);
  }

//  printf("ID: %lu, CPU: %d %d %d %d\n", pthread_self(), records[0], records[1], records[2], records[3]);
  return 0;
}

int main() {

//  int numberOfProcessors = sysconf(_SC_NPROCESSORS_ONLN);
//  printf("Number of processors: %d\n", numberOfProcessors);
  int numberOfThreads = 32;

  pthread_t threads[numberOfThreads];
  int args[numberOfThreads];

  pthread_attr_t attr;
  pthread_attr_init(&attr);

  for (int i = 0; i < numberOfThreads; i++) {

    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(i, &cpus);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

    args[i] = i;
    pthread_create(&threads[i], &attr, DoWork, &args[i]);
  }

//  int records[4];
//  for (int i = 0; i < 4; ++i) {
//    records[i] = sched_getcpu();
//    sleep(1);
//  }
//  printf("ID: %lu, CPU: %d %d %d %d\n", pthread_self(), records[0], records[1], records[2], records[3]);

  for (int i = 0; i < numberOfThreads; i++) {
    pthread_join(threads[i], NULL);
  }

  return 0;
}