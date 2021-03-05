#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <limits>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <bcl/containers/experimental/cuda/DArray.hpp>
// #include <curand.h>
// #include <curand_kernel.h>

template <typename T>
__global__ void kernel(size_t num_blocks, size_t transfer_size, size_t num_lookups,
            BCL::cuda::device_vector<size_t>& rand_nums,
            BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>>& buffer,
            BCL::cuda::DArray<T>& array) {
  size_t ltid = threadIdx.x;
  size_t local_extent = blockDim.x;
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t warp_id = tid / 32;
  size_t block_id = blockIdx.x;

  if (block_id >= num_blocks) {
    return;
  }

  for (size_t i = 0; i < num_lookups; i++) {
    size_t dest_rank = rand_nums[block_id + i*num_blocks*2] % array.d_ptrs_.size();

    size_t rand_loc = rand_nums[block_id+num_blocks + i*num_blocks*2] % (array.local_size() - transfer_size);

    auto ptr = array.d_ptrs_[dest_rank].get() + rand_loc;

    T* buf = buffer.data() + block_id * transfer_size;
    BCL::cuda::memcpy_warp(buf, ptr, transfer_size*sizeof(T));
    BCL::cuda::flush();
  }
}

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
  size_t threads_per_block = 32;
  size_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
  // 3b) Number of lookups to perform, per block
  size_t num_lookups = 20;

  BCL::print("Creating random device vector...\n");
  auto rand_nums = random_device_vector<size_t>(num_blocks*2*num_lookups);

  size_t global_size = global_data_size / sizeof(T);
  size_t transfer_size = transfer_data_size / sizeof(T);

  assert(transfer_size > 0);

  BCL::print("Creating DArray...\n");
  BCL::cuda::DArray<T> array(global_size);
  BCL::print("Created DArray...\n");

  BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>> buffer(transfer_size * num_blocks);

  BCL::print("Creating buffer of size %lu\n", transfer_size * num_blocks);

  BCL::cuda::barrier();
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::print("Beginning operation...\n");

  kernel<<<num_blocks, threads_per_block>>>
           (num_blocks, transfer_size, num_lookups, rand_nums, buffer, array);

  cudaDeviceSynchronize();

  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  size_t transferred_data = transfer_data_size * num_blocks * num_lookups * BCL::nprocs();
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
