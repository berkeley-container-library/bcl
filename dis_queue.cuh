#include <cub/cub.cuh>

//works for only two GPUs
template<typename T>
struct DQueue {
    int my_pe;
    int n_pes;
    T *queue;
    uint32_t *local_end;
    int *end;
    
    uint32_t capacity;

    DQueue(int _my_pe, int _n_pes, uint32_t _capacity) {
	my_pe = _my_pe;
	n_pes = _n_pes;
  	capacity = _capacity;
	queue = (T *)nvshmem_malloc(sizeof(T)*capacity);
	CUDA_CHECK(cudaMemset(queue, 0, sizeof(T)*capacity));
	end = (int *)nvshmem_malloc(sizeof(int));
	CUDA_CHECK(cudaMemset(end, 0, sizeof(int)));
	CUDA_CHECK(cudaMallocManaged(&local_end, sizeof(uint32_t)));
	CUDA_CHECK(cudaMemset(local_end, 0, sizeof(int)));
    }

    void release()
    {
	 if(queue!=NULL)
	     nvshmem_free(queue);
	 if(end!=NULL)
	     nvshmem_free(end);
	 if(local_end!=NULL)
	     CUDA_CHECK(cudaFree(local_end));
    }

    __device__ void push_warp(T item, int pe)
    {
	 unsigned mask = __activemask();
	 int rank = __popc(mask & lanemask_lt());
	 int total = __popc(mask);
	 uint32_t alloc;
	 if(rank == 0)
	     alloc = atomicAdd(local_end, total);
	 alloc = __shfl_sync(mask, alloc, __ffs(mask)-1);
	 nvshmem_int_p((queue+alloc+rank), item, pe);
    }

    __forceinline__ __device__ int push_block_step0(int pe)
    {
	 extern __shared__ int s[];
	 int shareMem_offset = ((pe<my_pe)?pe:pe-1)*TOTAL_WARPS_BLOCK + WARPID_BLOCK;
	 unsigned int mask = __match_any_sync(__activemask(), pe);
	 uint32_t total = __popc(mask);
	 uint32_t rank = __popc(mask & lanemask_lt());
	 if(rank == 0)
	    s[shareMem_offset] = total;
	 return rank;
    }

    __forceinline__ __device__ void push_block_step1(int item, int pe, int rank)
    {
	 extern __shared__ int s[];
	 int shareMem_offset = ((pe<my_pe)?pe:pe-1)*TOTAL_WARPS_BLOCK + WARPID_BLOCK-1;
	 int dataPool_offset = TOTAL_WARPS_BLOCK*(n_pes-1) + ((pe<my_pe)?pe:pe-1)*1024;
	 int pool_offset;
	 if(WARPID_BLOCK == 0) pool_offset = 0;
	 else pool_offset = s[shareMem_offset];
	 s[dataPool_offset+pool_offset+rank] = item;
    }

    __device__ void print_shareMem(int shared_size)
    {
	    extern __shared__ int s[];
	    if(threadIdx.x ==0)
            {
		 printf("shared size %d\n", shared_size/4);
	    for(int i=0; i<shared_size/4; i++)
	    {
		    if(i == 20)
			   printf("----------------------\n");
		    printf("%d\n", s[i]);
	    }
	    }
    }

    __forceinline__ __device__ void push_block(bool ifpush, int pe, T item)
    {
	 typedef cub::WarpScan<int> WarpScan;
	 extern __shared__ int s[];
	 int rank;
	 if(ifpush)
	     rank = push_block_step0(pe); 
	 __syncthreads();
	 __shared__ typename WarpScan::TempStorage temp_storage[32];
	 if((WARPID_BLOCK) < (n_pes-1))
	 {
	     int thread_data = 0;
	     if(LANE_ < TOTAL_WARPS_BLOCK)
		  thread_data = s[WARPID_BLOCK*TOTAL_WARPS_BLOCK + LANE_];
	     __syncwarp();
	     WarpScan(temp_storage[WARPID_BLOCK]).InclusiveSum(thread_data, thread_data);
	     if(LANE_ < TOTAL_WARPS_BLOCK)
	     {
	     	s[WARPID_BLOCK*TOTAL_WARPS_BLOCK + LANE_] = thread_data;
	     }
	 }
	 __syncthreads();
	 if(ifpush)
	     push_block_step1(item, pe, rank);
	 __syncthreads();
	 for(int i=0; i<n_pes-1; i++)
	 {
             __shared__ uint32_t alloc;
	     int send_size = s[(i+1)*TOTAL_WARPS_BLOCK-1];
	     if(send_size >0)
	     {
	        if(threadIdx.x == 0)
	     	   alloc = atomicAdd(local_end+i, send_size);
	        __syncthreads();
		int send_pe = (i<my_pe)?i:i+1;
	        nvshmemx_int_put_block((int *)queue+alloc, s+TOTAL_WARPS_BLOCK*(n_pes-1)+i*1024, send_size, send_pe);
	     }
	     if(LANE_ < TOTAL_WARPS_BLOCK)
		s[i*TOTAL_WARPS_BLOCK + LANE_] = 0;
	 }
    }
};



