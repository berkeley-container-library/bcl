#include <bcl/bcl.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include "bcl/containers/experimental/arh/arh.hpp"

bool thread_run = true;

std::vector<std::atomic<size_t>> issueds(16);
std::vector<std::atomic<size_t>> receiveds(16);
size_t req_num;
size_t rep_num;
ARH::ThreadBarrier threadBarrier;
const size_t payload_size = 4080; // ~4KiB
char payload[payload_size];

void empty_handler(gex_Token_t token, void *buf, size_t nbytes, gex_AM_Arg_t id) {
  {

  }
  gex_AM_ReplyMedium(token, rep_num, buf, nbytes, GEX_EVENT_NOW, 0, id);
}

void reply_handler(gex_Token_t token, void *buf, size_t nbytes, gex_AM_Arg_t id) {
  receiveds[id]++;
}

void worker(int id) {
  size_t num_ams = 10000;
  ARH::SimpleTimer timer;

  srand48(BCL::rank());
  threadBarrier.wait();
  if (id == 0) {
    BCL::barrier();
  }
  threadBarrier.wait();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();
//    size_t remote_proc = lrand48() % (BCL::nprocs() - 1);
//    if (remote_proc >= BCL::rank()) {
//      remote_proc++;
//    }

    issueds[id]++;
    int rv = gex_AM_RequestMedium(BCL::tm, remote_proc, req_num, payload, payload_size, GEX_EVENT_NOW, 0, id);
  }

  while (receiveds[id] < issueds[id]) {
    gasnet_AMPoll();
  }
  threadBarrier.wait();
  if (id == 0) {
    BCL::barrier();
  }
  threadBarrier.wait();
  auto end = std::chrono::high_resolution_clock::now();
  double duration_s = std::chrono::duration<double>(end - begin).count();

  double bandwidth_node_s = payload_size * num_ams * 32 / duration_s;
  if (id == 0) {
    BCL::print("bandwidth = %.3lf MB/S\n", bandwidth_node_s / 1e6);
  }
}

int main() {
  BCL::init();
#ifdef GASNETC_GNI_MULTI_DOMAIN
  if (BCL::rank() == 0)
    std::printf("enable gasnet multi domain\n");
#endif
  size_t max_args = gex_AM_MaxArgs();
  size_t handler_num = GEX_AM_INDEX_BASE;
  req_num = handler_num++;
  rep_num = handler_num++;

  gex_AM_Entry_t entry[2];
  entry[0].gex_index = req_num;
  entry[0].gex_fnptr = (gex_AM_Fn_t) empty_handler;
  entry[0].gex_flags = GEX_FLAG_AM_MEDIUM | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 1;

  entry[1].gex_index = rep_num;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply_handler;
  entry[1].gex_flags = GEX_FLAG_AM_MEDIUM | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 1;

  int rv = gex_EP_RegisterHandlers(BCL::ep, entry, 2);

  threadBarrier.init(16);

  std::vector<std::thread> worker_pool;
  for (size_t i = 0; i < 16; ++i) {
    auto t = std::thread(worker, i);
    worker_pool.push_back(std::move(t));
  }

  for (size_t i = 0; i < 16; ++i) {
    worker_pool[i].join();
  }

  BCL::finalize();
  gasnet_exit(0);
  return 0;

}