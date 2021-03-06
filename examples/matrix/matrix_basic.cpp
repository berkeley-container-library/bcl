#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>

int main(int argc, char** argv) {
	// How big to make each process' shared segment, in MB.
	size_t segment_size = 1024;

	BCL::init(segment_size);

	printf("Hello, world! I am process %lu / %lu\n", BCL::rank(), BCL::nprocs());

  // Create a distributed matrix of size 1024 x 1024
	BCL::DMatrix<float> matrix({1024, 1024});

	BCL::finalize();

	return 0;
}