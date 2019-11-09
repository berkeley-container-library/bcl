#include <gasnetex.h>
#include <queue>
#include <mutex>
#include <thread>
#include <cassert>
#include "arh_thread_barrier.hpp"

// GASNet-EX context variables

gex_Client_t client;
gex_EP_t ep;
gex_TM_t tm;
const char* clientName = "BCL";

size_t rank, nprocs;

size_t num_threads = 16;

// Process-level variables for thread synchronization
std::vector<std::atomic<size_t>> issueds(num_threads);
std::vector<std::atomic<size_t>> receiveds(num_threads);
size_t req_num;
size_t rep_num;
ARH::ThreadBarrier threadBarrier;

void barrier() {
  gex_Event_t event = gex_Coll_BarrierNB(tm, 0);
  gex_Event_Wait(event);
}

void empty_handler(gex_Token_t token, gex_AM_Arg_t id) {
  {

  }
  gex_AM_ReplyShort(token, rep_num, 0, id);
}

void reply_handler(gex_Token_t token, gex_AM_Arg_t id) {
  receiveds[id]++;
}

void worker(int id) {
  size_t num_ams = 10000;

  srand48(rank*id + id);
  if (id == 0) {
    barrier();
  }
  threadBarrier.wait();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % nprocs;

    issueds[id]++;
    int rv = gex_AM_RequestShort(tm, remote_proc, req_num, 0, id);
    while (receiveds[id] < issueds[id]) {
      gasnet_AMPoll();
    }
  }

  if (id == 0) {
    barrier();
  }
  threadBarrier.wait();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6 * duration;
  double latency_us = duration_us / num_ams;
  if (id == 0 && rank == 0) {
    printf("%lf us/AM (%lf s total)\n", latency_us, duration);
  }
} 


int main() {
  gex_Client_Init(&client, &ep, &tm, clientName, NULL, NULL, 0);

    #ifndef GASNET_PAR
      printf("Need to use a par build of GASNet-EX.\n");
      assert(false);
    #endif

  rank = gex_System_QueryJobRank();
  nprocs = gex_System_QueryJobSize();

  size_t max_args = gex_AM_MaxArgs();
  size_t handler_num = GEX_AM_INDEX_BASE;
  req_num = handler_num++;
  rep_num = handler_num++;

  gex_AM_Entry_t entry[2];
  entry[0].gex_index = req_num;
  entry[0].gex_fnptr = (gex_AM_Fn_t) empty_handler;
  entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 1;

  entry[1].gex_index = rep_num;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply_handler;
  entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 1;

  int rv = gex_EP_RegisterHandlers(ep, entry, 2);

  threadBarrier.init(num_threads);

  std::vector<std::thread> worker_pool;
  for (size_t i = 0; i < num_threads; ++i) {
    auto t = std::thread(worker, i);
    worker_pool.push_back(std::move(t));
  }

  for (size_t i = 0; i < num_threads; ++i) {
    worker_pool[i].join();
  }

  return 0;
}
