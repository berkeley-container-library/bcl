#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <limits>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <bcl/containers/experimental/cuda/DArray.hpp>
// #include <curand.h>
// #include <curand_kernel.h>

template <typename T>
BCL::cuda::device_vector<T> random_device_vector(size_t n,
                                                 size_t extent = std::numeric_limits<size_t>::max()) {
  std::vector<T> v(n);
  srand48(BCL::rank());
  for (size_t i = 0; i < n; i++) {
    v[i] = lrand48() % extent;
  }
  return BCL::cuda::device_vector<T>(v.begin(), v.end());
}

/*
  Parameters are
    1) Global data size
    2) Transfer size
    3) Number of lookups
       a) Number of threads
       b) Lookups per thread
*/

template <typename T = int>
void run_kernel(size_t transfer_data_size) {
  BCL::print("Running kernel...\n");
  static_assert(std::is_same<T, int>::value);

  // XXX: Important parameters
  // 1) Global data size, in bytes.
  size_t global_data_size = size_t(2)*1024*1024*size_t(1024);
  // 3a) Number of threads to launch, per processor
  size_t num_threads = 1000;
  // 3b) Number of lookups to perform, per thread
  size_t num_lookups = 20;

  BCL::print("Creating random device vector...\n");
  auto rand_nums = random_device_vector<size_t>(num_threads*2*num_lookups);

  size_t global_size = global_data_size / sizeof(T);
  size_t transfer_size = transfer_data_size / sizeof(T);

  assert(transfer_size > 0);

  BCL::print("Creating DArray...\n");
  BCL::cuda::DArray<T> array(global_size);
  BCL::print("Created DArray...\n");

  BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>> buffer(transfer_size * num_threads);

  BCL::print("Creating buffer of size %lu\n", transfer_size * num_threads);

  BCL::cuda::barrier();
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::print("Beginning operation...\n");

  BCL::cuda::global_launch(num_threads*BCL::nprocs(),
            [] __device__ (auto info,
                           size_t transfer_size,
                           size_t num_lookups,
                           BCL::cuda::device_vector<size_t>& rand_nums,
                           BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>>& buffer,
                           BCL::cuda::DArray<T>& array) {
              for (size_t i = 0; i < num_lookups; i++) {
                /*
                if (info.ltid + i*info.local_extent*2 >= rand_nums.size()) {
                  printf("%lu out of range 1\n", info.tid);
                  return;
                }
                */
                size_t dest_rank = rand_nums[info.ltid + i*info.local_extent*2] % array.d_ptrs_.size();
                /*
                if (info.ltid+info.local_extent + i*info.local_extent*2 >= rand_nums.size()) {
                  printf("%lu out of range 2\n", info.tid);
                  return;
                }
                */
                size_t rand_loc = rand_nums[info.ltid+info.local_extent + i*info.local_extent*2] % (array.local_size() - transfer_size);
                /*
                if (dest_rank >= array.d_ptrs_.size()) {
                  printf("%lu out of range 3\n", info.tid);
                  return;
                }
                */
                auto ptr = array.d_ptrs_[dest_rank].get() + rand_loc;
                /*
                if (info.ltid * transfer_size + transfer_size > buffer.size()) {
                  printf("%lu out of range 4 %lu > %lu\n",
                         info.tid, info.ltid * transfer_size + transfer_size, buffer.size());
                  return;
                }
                */
                T* buf = buffer.data() + info.ltid * transfer_size;
                BCL::cuda::memcpy(buf, ptr, transfer_size*sizeof(T));
                BCL::cuda::flush();
              }
            }, transfer_size, num_lookups, rand_nums, buffer, array);

  cudaDeviceSynchronize();

  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  size_t transferred_data = transfer_data_size * num_threads * num_lookups * BCL::nprocs();
  double bw = 1.0e-9*(transferred_data / duration);
  double bwpn = bw / BCL::nprocs();
  BCL::print("Ended in %lf seconds.\n", duration);
  BCL::print("Total BW %lf GB/s\n", bw);
  BCL::print("%lf GB/s per process\n", bwpn);
}

int main(int argc, char** argv) {
  BCL::init(64);

  BCL::cuda::init(4096);

  if (argc < 2) {
    if (BCL::rank() == 0) {
      fprintf(stderr, "usage: ./irregular-lookup [transfer_data_size (bytes)]\n");
    }
    BCL::cuda::finalize();
    BCL::finalize();
    return 1;
  }

  BCL::print("Beginning experiment...\n");
  size_t transfer_data_size = size_t(256)*size_t(1024);
  transfer_data_size = std::atoll(argv[1]);
  run_kernel(transfer_data_size);

  BCL::cuda::finalize();

  BCL::finalize();
  return 0;
}
