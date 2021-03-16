#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <cstdio>
#include <futar/futar.hpp>

template <typename T, typename Allocator>
void print_vec(std::vector<T, Allocator>& v) {
	for (size_t i = 0; i < v.size(); i++) {
		printf(" %.2lf", v[i]);
	}
	printf("\n");
}

int main(int argc, char** argv) {
	// How big to make each process' shared segment, in MB.
	size_t segment_size = 1024;

	BCL::init(segment_size);

  // Create a distributed matrix of size 1024 x 1024
	BCL::DMatrix<float> matrix({32, 32}, BCL::BlockRow());

  // Print information about tile size and distribution.
  if (BCL::rank() == 0) {
  	printf("Just created a %lu x %lu matrix.  Here's some info:\n",
  		     matrix.shape()[0], matrix.shape()[1]);
	  matrix.print_info();
  }

  // Apply the matrix to random values.
  srand48(BCL::rank());
  matrix.apply_inplace([](float a) { return drand48(); });

  // Create future pool to hold/handle all the asynchronous operations we're
  // going to issue.  We choose a capacity of 10, which means that there will
  // be no more than 10+1 outstanding futures at any one time.
  // (Technically here, each of the futures in the future pool contains two
  //  asynchronous get futures, so that's 20+2 outstanding asynchronous gets.)
  size_t capacity = 10;
  futar::FuturePool<float> pool(capacity);

  // Barrier necessary to ensure all processes are finished
  // before rank 0 reads.
  BCL::barrier();

  if (BCL::rank() == 0) {
    // Let's get each row of the matrix.
    for (size_t i = 0; i < matrix.shape()[0]-1; i++) {
    	// Issue asynchronous get to row `i`
		  size_t tile_num = i / matrix.tile_shape()[0];
		  size_t row_num = i % matrix.tile_shape()[0];
		  auto row_buf_1 = matrix.arget_tile_row(tile_num, 0, row_num);

      // Issue asynchronous get to row `i+1`
		  tile_num = (i+1) / matrix.tile_shape()[0];
		  row_num = (i+1) % matrix.tile_shape()[0];
		  auto row_buf_2 = matrix.arget_tile_row(tile_num, 0, row_num);

      // Note that the vectors returned by `arget_tile_row` use the
      // bcl_allocator by default, not std::allocator. This is because
      // it's faster to do a copy into pinned memory.
		  using vector_type = std::vector<float, BCL::bcl_allocator<float>>;

      auto dot_prod_future =
		  futar::with_future([](vector_type row1, vector_type row2) {
		  	                   float sum = 0;
		  	                   for (size_t i = 0; i < row1.size(); i++) {
		  	                     sum += row1[i] * row2[i];
		  	                   }
		  	                   return sum;
		  	                 },
		  	                 std::move(row_buf_1), std::move(row_buf_2));

		  pool.push_back(std::move(dot_prod_future));
	  }
  }

  pool.drain();

  while (pool.size() > 0) {
  	float dot_prod = pool.get();
  	std::cout << dot_prod << std::endl;
  }

	BCL::finalize();

	return 0;
}