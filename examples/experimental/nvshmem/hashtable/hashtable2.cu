#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/DArray.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <bcl/containers/experimental/cuda/DHashMap.cuh>

__global__ void testcheck(BCL::cuda::DHashMap<uint32_t, int> map, size_t size, int my_pe)
{
  if(TID == 0)
  {
    uint32_t sum=0;
    auto ptr = map.flags_[my_pe].get().rptr();
    for(int i=0; i<map.local_capacity_; i++)
     if(ptr[i] == 1) sum++;

    printf("PE %d, sum %d, local_capacity %d\n", my_pe, sum, map.local_capacity_);
  }
}

int main(int argc, char** argv) {
  BCL::init(64);
  BCL::cuda::init(1024);
  std::cout << "npes: "<< BCL::nprocs() << std::endl;
  BCL::cuda::DHashMap<uint32_t, int> map(1024);

  int file_size = 49;
  if(BCL::rank() == 1)
    file_size = 50;
  BCL::cuda::HashMapEntry<uint32_t, int> * rand;
  cudaMallocManaged(&rand, sizeof(BCL::cuda::HashMapEntry<uint32_t, int>)*file_size);
  uint32_t seed = 98735*BCL::rank();
  std::srand(seed);
  for(int i=0; i<file_size; i++)
  {
    uint32_t r = std::rand();
    BCL::cuda::HashMapEntry<uint32_t, int> rr{r,r};
    rand[i] = rr;
  }

  BCL::cuda::barrier();
  BCL::cuda::launch(file_size*32, [] __device__ (size_t tid, BCL::cuda::HashMapEntry<uint32_t, int> * rand, BCL::cuda::DHashMap<uint32_t, int> &map, int pe)
                              {
                                BCL::cuda::HashMapEntry<uint32_t, int> tuple = rand[WARPID];
                                bool res = map.insert_warp(tuple.key, tuple.value, pe);
                                if(res == false && LANE==0)
                                  printf("fail to insert\n");
                              }, rand, map, BCL::rank());
  cudaDeviceSynchronize();
  BCL::cuda::barrier();

  testcheck<<<1,1>>>(map, 512, BCL::rank());
  cudaDeviceSynchronize();
  BCL::cuda::barrier();

  

//  if (BCL::rank() == 0) {
//    BCL::cuda::launch(1, [] __device__ (size_t tid, BCL::cuda::DHashMap<int, int>& map) {
//                           for (size_t i = 0; i < 10; i++) {
//                             map.insert(i, i);
//                           }
//                         }, map);
//    cudaDeviceSynchronize();
//  }
//  BCL::cuda::barrier();
//
//  if (BCL::rank() == 1) {
//    BCL::cuda::launch(1, [] __device__ (size_t tid, BCL::cuda::DHashMap<int, int>& map) {
//                           for (size_t i = 0; i < 10; i++) {
//                             int value = map.find(i);
//                             printf("{%lu, %d}\n", i, value);
//                           }
//                         }, map);
//    cudaDeviceSynchronize();
//  }
//
  BCL::cuda::finalize();
  BCL::finalize();
  return 0;
}
