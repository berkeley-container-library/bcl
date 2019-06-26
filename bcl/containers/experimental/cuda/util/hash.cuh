template<typename T>  
struct MyHash{
  __device__ __host__ uint32_t operator()(T key, uint32_t seed)
  {
    return key;
  }
};
