#include<iostream>
#include<string>

#include<cuda.h>

#include "nvshmem.h"
#include "nvshmemx.h"

#include "util/error_util.cuh"
#include "util/time.cuh"
#include "util/util.cuh"

#include "test.cuh"

using namespace std;

__device__ int clockrate;

int main(int argc, char *argv[]) 
{
    uint32_t num_items = (1<<28); 
    float local_percentage = 0.5;
    if(argc == 1)
    {
       cout<<"./test -n <number of items going to insert=(1<<28)> -l <percentage item-inserts are local=0.5>\n "; 
    }
    if(argc > 1)
        for(int i=1; i<argc; i++) {
            if(string(argv[i]) == "-n")
                num_items = stoi(argv[i+1]);
            if(string(argv[i]) == "-l")
                local_percentage = stof(argv[i+1]);
        }
    
    nvshmem_init();
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int numBlock = 224;
    int numThread = 256;
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)_runQueueInsert_block2<int, uint32_t, uint32_t>);
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,(void *)_launchWarpPer32Items4<int, int, PageRankFuncWarp<int,int,float>, PageRank<int, int, float>, Context<PageRankEntry<int, float>>>, numThread, 0);
    cout << "n_pes: "<< n_pes << " my_pe: "<< my_pe << " numBlocks: "<< numBlock << " numThreads: "<< numThread << " number of items to insert: "<< num_items << " local percentage: " << local_percentage << endl;

    cudaDeviceProp prop;
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(my_pe));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, my_pe%dev_count));
    cout << "PE: "<< my_pe << " deviceCount " << dev_count << " device name " << prop.name << " number of SMs: "<< prop.multiProcessorCount << endl;
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0, cudaMemcpyHostToDevice));

    uint32_t max_rand = (1<<30);
    uint32_t *items;
    CUDA_CHECK(cudaMallocManaged(&items, num_items*sizeof(uint32_t)));
    srand(12321);
    for(int i=0; i<num_items; i++)
	items[i] = rand()%max_rand;

    DQueue<int> queue(my_pe,n_pes, num_items);

    nvshmem_barrier_all();
    float time = 0.0;
    GpuTimer timer;
    cout << "PE " << my_pe <<" About to start the insertion\n";
    nvshmem_barrier_all();
    timer.Start();
    runQueueInsert_block2(numBlock, numThread, queue, items, num_items, max_rand, local_percentage);
//    runQueueInsert(numBlock, numThread, queue, items, num_items, max_rand, local_percentage);
    timer.Stop();
    nvshmem_barrier_all();
    int end_h;
    CUDA_CHECK(cudaMemcpy(&end_h, queue.end, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "PE "<< my_pe << " receive " << end_h <<  " items\n";
    check_result<<<numBlock, numThread>>>(queue, 12321);
    CUDA_CHECK(cudaDeviceSynchronize());
    time = timer.ElapsedMillis();
    CUDA_CHECK(cudaFree(items));
    nvshmem_barrier_all();
    for(int i=0; i<n_pes; i++)
    {
        nvshmem_barrier_all();
	if(my_pe == i)
    	std::cout << "End program "<< my_pe << " time: "<< time << " throughput: "<< *(queue.local_end)/(time/1000)/1000000<< "M/s" << std::endl;
    }
    nvshmem_finalize();
    return 0;
}
