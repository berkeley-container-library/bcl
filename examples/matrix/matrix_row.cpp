// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>

int main(int argc, char** argv) {
	// How big to make each process' shared segment, in MB.
	size_t segment_size = 1024;

	BCL::init(segment_size);

  // Create a distributed matrix of size 1024 x 1024
	BCL::DMatrix<float> matrix({1024, 1024}, BCL::BlockRow());
	// This would mean each row block has a fixed height of 10
	// BCL::DMatrix<float> matrix({1024, 1024}, BCL::BlockRow({10, BCL::Tile::div}));

  // Print information about tile size and distribution.
  if (BCL::rank() == 0) {
  	printf("Just created a %lu x %lu matrix.  Here's some info:\n",
  		     matrix.shape()[0], matrix.shape()[1]);
	  matrix.print_info();
  }

	BCL::finalize();

	return 0;
}