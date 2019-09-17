#include "dis_queue.cuh"

template<typename T, typename I, typename S>
__global__ void _runQueueInsert(DQueue<T> queue, I* items, S size, uint32_t cut_line)
{
     for(uint32_t i = TID; i<size; i=i+gridDim.x*blockDim.x)
     {
	 if(items[i] >= cut_line)
	     queue.push_warp(12321, (queue.my_pe)^1);
	 __syncwarp();
     }
}

template<typename T>
__global__ void _update_end(DQueue<T> queue)
{
      nvshmem_int_put(queue.end, (int *)queue.local_end, sizeof(int), (queue.my_pe)^1);
}

template<typename T>
__global__ void check_result(DQueue<T> queue, int item)
{
      for(int i=TID; i< *queue.end; i=i+gridDim.x*blockDim.x)
	  if(queue.queue[i] != item)
	       printf("ERROR: %d item %d is not %d\n", i, queue.queue[i], item);
}

template<typename T, typename I, typename S>
__global__ void _runQueueInsert_block(DQueue<T> queue, I *array, S size, uint32_t cut_line)
{
    for(int i = TID; i-LANE_<size; i=i+TOTAL_THREADS)
    {
	 queue.push_block((array[i]>=cut_line), (queue.my_pe^1), 12321);
    }
}

template<typename T, typename I, typename S>
__global__ void _runQueueInsert_block2(DQueue<T> queue, I *array, S size, uint32_t cut_line)
{
    for(int i = TID; i-LANE_<size; i=i+TOTAL_THREADS)
    {
	 queue.push_block2((array[i]>=cut_line), (queue.my_pe^1), 12321);
    }
}


template<typename T, typename I, typename S>
void runQueueInsert(int numBlock, int numThread, DQueue<T> queue, I * items, S size, uint32_t max_rand, float local_percentage)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	_runQueueInsert<<<numBlock, numThread, 0, stream>>>(queue, items, size, uint32_t(max_rand*local_percentage));
	_update_end<<<1,1,0,stream>>>(queue);
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename T, typename I, typename S>
void runQueueInsert_block(int numBlock, int numThread, DQueue<T> queue, I * items, S size, uint32_t max_rand, float local_percentage)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	int shared_size = (queue.n_pes-1)*(numThread>>5)*sizeof(int)+(queue.n_pes-1)*1024*sizeof(T);
	_runQueueInsert_block<<<numBlock, numThread, shared_size, stream>>>(queue, items, size, uint32_t(max_rand*local_percentage));
	_update_end<<<1,1,0,stream>>>(queue);
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename T, typename I, typename S>
void runQueueInsert_block2(int numBlock, int numThread, DQueue<T> queue, I * items, S size, uint32_t max_rand, float local_percentage)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	int shared_size = (queue.n_pes-1)*sizeof(int)+(queue.n_pes-1)*1024*sizeof(T);
	_runQueueInsert_block2<<<numBlock, numThread, shared_size, stream>>>(queue, items, size, uint32_t(max_rand*local_percentage));
	_update_end<<<1,1,0,stream>>>(queue);
	CUDA_CHECK(cudaStreamSynchronize(stream));
}
