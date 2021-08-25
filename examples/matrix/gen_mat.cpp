// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <bcl/containers/SPMatrix.hpp>
#include <bcl/containers/algorithms/experimental_gemm.hpp>
#include <bcl/containers/algorithms/spgemm.hpp>

#include "generate_spmat.hpp"

#include <chrono>

int main(int argc, char** argv) {
	// How big to make each process' shared segment, in MB.
	size_t segment_size = 2048;

	BCL::init(segment_size);

  size_t m, k, n;

  // parameters: [number of samples] [number of categories] [embedding size] [nnz_row] [alpha]

	if (argc >= 4) {
		m = std::atoi(argv[1]);
		k = std::atoi(argv[2]);
		n = std::atoi(argv[3]);
	} else {
		BCL::finalize();
		return 1;
	}

  size_t nnz_per_row = 100;
  double alpha = 0.0;

	if (argc >= 5) {
		nnz_per_row = std::atoi(argv[4]);
	}

	if (argc >= 6) {
		alpha = std::atof(argv[5]);
	}

  using value_type = float;
	using index_type = long long int;

	BCL::print("Generating blocks...\n");

  auto blocks = BCL::block_matmul(m, n, k);

  srand48(BCL::rank());
  BCL::print("Generating matrix (%lu x %lu), alpha %lf, nnz_per_row %lu\n",
  	         m, k, alpha, nnz_per_row);
  auto a = BCL::generate_matrix<value_type, index_type>(m, k, nnz_per_row, alpha, BCL::NewBlockRow{});

	BCL::DMatrix<value_type> b({k, n}, BCL::NewBlockRow{});
	BCL::DMatrix<value_type> c({m, n}, BCL::NewBlockRow{});

	BCL::print("Generated A (%lu x %lu matrix) with %lu nnz\n",
		         a.shape()[0], a.shape()[1], a.nnz());

	BCL::print("Multipyling by B (%lu x %lu dense matrix)\n",
		         b.shape()[0], b.shape()[1]);

	BCL::print("To produce C (%lu x %lu dense matrix)\n",
		         c.shape()[0], c.shape()[1]);

/*
  if (BCL::rank() == 0) {
  	printf("A:\n");
		a.print_details();
  	printf("B:\n");
		b.print_details();
  	printf("C:\n");
		c.print_details();
  }
  */

	size_t real_nnz = a.count_nonzeros_();

	BCL::print("Counted %lu real nonzeros\n", real_nnz);

	b = 1;
	c = 0;

	size_t cache_size = 128*1024*1024;

  BCL::barrier();
	auto begin = std::chrono::high_resolution_clock::now();
	// BCL::rowwise_gemm(a, b, c);
	// BCL::cached_rowwise_gemm(a, b, c, cache_size);
	BCL::batched_rowwise_gemm(a, b, c);
	BCL::barrier();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double>(end - begin).count();

	BCL::barrier();

	BCL::print("Matrix Multiply took %lf s\n", duration);

	BCL::print("Comm/comp %lf / %lf\n", BCL::row_comm, duration - BCL::row_comm);

	BCL::print("Sum is %lf\n", c.sum());

	BCL::finalize();

	return 0;
}
