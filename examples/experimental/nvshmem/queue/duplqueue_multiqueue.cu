#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/DuplQueue.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <chrono>

int main(int argc, char** argv) {
  BCL::init(16);

  printf("Hello, world! I am rank %lu/%lu\n",
         BCL::rank(), BCL::nprocs());
  fflush(stdout);
  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  BCL::cuda::init();

  size_t insert_size = 8192;
  size_t inserts_per_proc = 8*1024;

  size_t num_inserts = inserts_per_proc*BCL::nprocs();

  size_t total_inserts_per_proc = inserts_per_proc*insert_size;
  size_t total_inserts = num_inserts*insert_size;

  BCL::print("Creating queues...\n");
  std::vector<BCL::cuda::DuplQueue<int>> queues;
  queues.reserve(BCL::nprocs());
  for (size_t i = 0; i < BCL::nprocs(); i++) {
    queues.push_back(BCL::cuda::DuplQueue<int>(i, total_inserts_per_proc*2));
  }

  BCL::print("Pushing queues to GPU...\n");
  BCL::cuda::device_vector<BCL::cuda::DuplQueue<int>> d_queues(BCL::nprocs());
  d_queues.assign(queues.begin(), queues.end());

  BCL::print("Creating random numbers...\n");
  BCL::cuda::device_vector<int> d_rand_nums(inserts_per_proc);
  std::vector<int> rand_nums(inserts_per_proc);

  srand48(BCL::rank());
  for (size_t i = 0; i < inserts_per_proc; i++) {
    rand_nums[i] = lrand48();
  }

  d_rand_nums.assign(rand_nums.begin(), rand_nums.end());

  BCL::print("Sending random numbers to GPU...\n");
  BCL::cuda::device_vector<int, BCL::cuda::bcl_allocator<int>> values(insert_size);
  // BCL::cuda::device_vector<int> values(insert_size);
  std::vector<int> values_local(insert_size, BCL::rank());
  values.assign(values_local.begin(), values_local.end());

  BCL::cuda::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  BCL::print("Pushing...\n");
  BCL::cuda::global_launch(num_inserts,
                     [] __device__ (size_t idx, BCL::cuda::device_vector<BCL::cuda::DuplQueue<int>>& queues,
                                    BCL::cuda::device_vector<int>& rand_nums,
                                    BCL::cuda::device_vector<int, BCL::cuda::bcl_allocator<int>>& values) {
                                    // BCL::cuda::device_vector<int>& values) {
                       // bool success = queues.data()[0].push(values.data(), values.size());
                       size_t queue = rand_nums[idx] % BCL::cuda::nprocs();
                       bool success = queues.data()[queue].push(values.data(), values.size());
                       if (!success) {
                         printf("AGH! I have failed!\n");
                       }
                     }, d_queues, d_rand_nums, values);

  BCL::print("synchronizing 1...\n");
  cudaDeviceSynchronize();
  BCL::print("synchronizing 2...\n");
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double data_moved = total_inserts*sizeof(int);

  double bw = data_moved / duration;
  double bw_gb = bw*1e-9;

  BCL::print("Total %lf s (%lf GB/s)\n", duration, bw_gb);

  std::vector<int> histogram_local(BCL::nprocs(), 0);
  BCL::cuda::device_vector<int> histogram(BCL::nprocs());
  histogram.assign(histogram_local.begin(), histogram_local.end());
  BCL::print("Popping...\n");
  BCL::cuda::launch(total_inserts,
                    [] __device__ (size_t idx, BCL::cuda::device_vector<BCL::cuda::DuplQueue<int>>& queues,
                                   BCL::cuda::device_vector<int>& histogram) {
                      int value = 12;
                      bool success = queues.data()[BCL::cuda::rank()].local_pop(value);
                      if (success && value >= 0 && value < BCL::cuda::nprocs()) {
                        atomicAdd(&histogram.data()[value], 1);
                      }
                    }, d_queues, histogram);
  cudaDeviceSynchronize();
  BCL::cuda::barrier();

  size_t total_counted = 0;
  for (size_t i = 0; i < histogram.size(); i++) {
    int hval = histogram[i];
    if (BCL::rank() == 0) {
      printf("%lu sees histogram[%lu] = %d\n", BCL::rank(), i, hval);
    }
    hval = BCL::allreduce<int>(hval, std::plus<int>{});
    BCL::print("%lu: %d\n", i, hval);
    total_counted += hval;
  }
  BCL::print("Counted %lu, inserts %lu\n",
         total_counted, total_inserts);
  assert(total_counted == total_inserts);

  BCL::cuda::barrier();

  BCL::print("Finished...\n");

  BCL::finalize();
  return 0;
}
