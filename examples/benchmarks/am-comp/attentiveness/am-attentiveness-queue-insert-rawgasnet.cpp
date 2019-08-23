#include <bcl/bcl.hpp>
#include <queue>
#include <mutex>

bool thread_run = true;

void service_ampoll() {
  while (thread_run) {
    gasnet_AMPoll();
  }
}

size_t issued = 0;
std::atomic<size_t> received = 0;

std::mutex mutex;
std::queue<int> queue;

void queue_insert_handler(gex_Token_t token, gex_AM_Arg_t value) {
  {
    std::lock_guard<std::mutex> guard(mutex);
    queue.push(value);
  }
  gex_AM_ReplyShort(token, GEX_AM_INDEX_BASE+1, 0);
}

void reply_handler(gex_Token_t token) {
  received++;
}

int main(int argc, char** argv) {
  size_t num_ams = 50000;

  BCL::init();
  BCL::print("Start!\n");

  size_t handler_num = GEX_AM_INDEX_BASE;
  size_t max_args = gex_AM_MaxArgs();

  gex_AM_Entry_t entry[2];
  entry[0].gex_index = handler_num++;
  entry[0].gex_fnptr = (gex_AM_Fn_t) queue_insert_handler;
  entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 1;

  entry[1].gex_index = handler_num++;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply_handler;
  entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 0;

  int rv = gex_EP_RegisterHandlers(BCL::ep, entry, 2);

  auto thread_ampool = std::thread(service_ampoll);

  srand48(BCL::rank());
  fprintf(stderr, "Starting experiment...\n");
  BCL::barrier();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();

    issued++;
    int rv = gex_AM_RequestShort(BCL::tm, remote_proc, GEX_AM_INDEX_BASE, 0, BCL::rank());
  }
  fprintf(stderr, "Waiting until finished...\n");
  while (received < issued) {}

  BCL::barrier();
  fprintf(stderr, "After barrier...\n");

  thread_run = false;
  thread_ampool.join();

  BCL::finalize();
  return 0;
}
