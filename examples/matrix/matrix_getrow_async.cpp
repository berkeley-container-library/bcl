#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <cstdio>

template <typename T>
void print_vec(std::vector<T>& v) {
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

  // Barrier necessary to ensure all processes are finished
  // before rank 0 reads.
  BCL::barrier();

  if (BCL::rank() == 0) {
    // Let's get each row of the matrix.
    for (size_t i = 0; i < matrix.shape()[0]; i++) {
		  size_t tile_num = i / matrix.tile_shape()[0];
		  size_t row_num = i % matrix.tile_shape()[0];
		  printf("Getting row %lu, which should be row %lu within tile %lu:\n",
		  	     i, row_num, tile_num);
		  auto row_buf = matrix.arget_tile_row(tile_num, 0, row_num);
		  row = row_buf.get();
		  print_vec(row);
	  }
  }

	BCL::finalize();

	return 0;
}