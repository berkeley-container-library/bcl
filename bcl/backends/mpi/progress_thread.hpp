#pragma once

#include <vector>
#include <list>
#include <thread>
#include <future>
#include <algorithm>
#include <type_traits>
#include <unistd.h>

#include "../../containers/Container.hpp"

#include "backend.hpp"

namespace BCL {
  std::thread prog_thread_handle;
  bool progress_running;

  extern MPI_Comm comm;

  extern void handle_rpc(int source, uint64_t exec_ptr, uint64_t fn_ptr,
    uint64_t args_size);

  namespace signal {
    const int progress_tag          =  0xDECAF;
    const int kill_progress         = 0x137DEA;
    const int rpc_args              = 0x137A45;
    const int rpc_rval              = 0x137435;

    const uint64_t rpc_label        = 0x137D00;
  }

  void progress_thread() {
    const size_t msg_size = 8;
    std::list <std::future <void>> futures;
    // TODO: Handle race when a rank simultaneously issues multiple RPCs
    // (only possible with threading)
    MPI_Request request;
    uint64_t msg[msg_size];
    uint64_t msg_code = 0;
    MPI_Irecv(msg, msg_size, MPI_UNSIGNED_LONG_LONG, MPI_ANY_SOURCE,
      BCL::signal::progress_tag, BCL::comm, &request);
    do {
      int flag;
      MPI_Status status;
      MPI_Test(&request, &flag, &status);
      if (flag) {
        int source = status.MPI_SOURCE;
        int count;
        MPI_Get_count(&status, MPI_UNSIGNED_LONG_LONG, &count);
        msg_code = msg[0];
        if (msg_code == BCL::signal::rpc_label) {
          uint64_t exec_ptr = msg[1];
          uint64_t fn_ptr = msg[2];
          uint64_t args_size = msg[3];
          futures.push_back(std::async(std::launch::async | std::launch::deferred,
            BCL::handle_rpc, source, exec_ptr, fn_ptr, args_size));
          MPI_Irecv(msg, msg_size, MPI_UNSIGNED_LONG_LONG, MPI_ANY_SOURCE,
            BCL::signal::progress_tag, BCL::comm, &request);
        } else if (msg_code == BCL::signal::kill_progress) {
        } else {
          throw std::runtime_error("Progress Engine: picked up junk message");
        }
      } else {
        futures.remove_if([] (auto &future) {
          return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        });
        usleep(1);
      }
    } while (msg_code != BCL::signal::kill_progress);

    std::for_each(futures.begin(), futures.end(), [] (auto &future) {
      future.wait();
    });
  }

  void start_progress_thread() {
    if (BCL::rank() == 0) {
      fprintf(stderr, "warning: you have enabled the progress thread, which is experimental.\n");
    }
    if (!BCL::progress_running) {
      BCL::progress_running = true;
      prog_thread_handle = std::thread(&progress_thread);
    }
  }

  void stop_progress_thread() {
    if (BCL::progress_running) {
      uint64_t request[1];
      request[0] = BCL::signal::kill_progress;
      MPI_Send(&request, 1, MPI_UNSIGNED_LONG_LONG, BCL::rank(),
        BCL::signal::progress_tag, BCL::comm);
      prog_thread_handle.join();
    }
  }
}
