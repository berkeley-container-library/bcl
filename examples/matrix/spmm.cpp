// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <bcl/containers/SPMatrix.hpp>
#include <bcl/containers/algorithms/experimental_gemm.hpp>
#include <bcl/containers/algorithms/spgemm.hpp>

int main(int argc, char** argv) {
	// How big to make each process' shared segment, in MB.
	size_t segment_size = 2048;

	BCL::init(segment_size);

  // Number of columns in the matrix B
	size_t num_columns = 1024;

  std::string fname = "";
	if (argc >= 2) {
		fname = argv[1];
	} else {
		BCL::finalize();
		return 1;
	}

	if (argc >= 3) {
		num_columns = std::atoi(argv[2]);
	}

  using value_type = float;
	using index_type = long long int;

	BCL::print("Multiplying matrix \"%s\" by %lu columns\n",
		         fname.c_str(), num_columns);


  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape.shape[0];
  size_t k = matrix_shape.shape[1];
  size_t n = num_columns;

  auto blocks = BCL::block_matmul(m, n, k);

  BCL::SPMatrix<value_type, index_type> a(fname, std::move(blocks[0]));

	BCL::DMatrix<value_type> b({k, n}, std::move(blocks[1]));
	BCL::DMatrix<value_type> c({k, n}, std::move(blocks[2]));

	b = 1;
	c = 0;

  BCL::print("Multiplying...\n");
	BCL::barrier();
	auto begin = std::chrono::high_resolution_clock::now();
	BCL::gemm(a, b, c);
	BCL::barrier();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double>(end - begin).count();

	BCL::print("Matrix Multiply took %lf s\n", duration);

	BCL::print("Sum is %lf\n", c.sum());

	BCL::finalize();

	return 0;
}
