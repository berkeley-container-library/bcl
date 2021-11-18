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
	bool threaded = false;

	BCL::init(segment_size, threaded);

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
  auto a = BCL::generate_matrix<value_type, index_type>(m, k, nnz_per_row, alpha, blocks[0]);

	BCL::DMatrix<value_type> b({k, n}, blocks[1]);
	BCL::DMatrix<value_type> c({m, n}, blocks[2]);

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

	size_t number_of_samples = 1;

	std::vector<double> times;

	for (size_t i = 0; i < number_of_samples; i++) {
		BCL::num_fetches = 0;
		BCL::row_comm = 0;
		b = 1;
		c = 0;

		size_t cache_size = 128*1024*1024;

	  BCL::barrier();
		auto begin = std::chrono::high_resolution_clock::now();
		BCL::gemm(a, b, c);
		BCL::barrier();
		auto end = std::chrono::high_resolution_clock::now();
		double duration = std::chrono::duration<double>(end - begin).count();

		BCL::barrier();

		BCL::print("Matrix Multiply took %lf s\n", duration);

		BCL::print("Comm/comp %lf / %lf\n", BCL::row_comm, duration - BCL::row_comm);

		BCL::print("Sum is %lf\n", c.sum());

		size_t bytes_fetched = BCL::num_fetches*n*sizeof(value_type);
		double gb_s = (1.0e-9*bytes_fetched) / BCL::row_comm;
		fprintf(stderr, "(%lu) %lf GB/s %lu bytes fetched from in %lf seconds\n",
			      BCL::rank(), gb_s, bytes_fetched, BCL::row_comm);
    fflush(stderr);
		times.push_back(duration);
	}

	BCL::barrier();
	fflush(stdout);
	fflush(stderr);
	BCL::barrier();
	usleep(10);
	BCL::barrier();

	std::sort(times.begin(), times.end());

	BCL::print("Matrix Multiply took %lf s (median)\n", times[times.size()/2]);

	size_t total_lookups = a.nnz();
	size_t lookup_bytes = sizeof(value_type)*b.shape()[1];
	double gb = 1e-9*total_lookups*lookup_bytes;
	double gb_s = gb / times[times.size() / 2];

	size_t bytes_fetched = BCL::num_fetches*n*sizeof(value_type);
	size_t actual_lookup_bytes = BCL::allreduce<size_t>(bytes_fetched, std::plus<size_t>{});
	double actual_gb = 1e-9*actual_lookup_bytes;
	double actual_gb_s = actual_gb / BCL::row_comm;

	BCL::print("%lu lookups of %lu bytes (%lf GB/s) (~%lu actual lookups for %lf GB/s [%lf / proc])\n",
		         total_lookups, lookup_bytes, gb_s,
		         actual_lookup_bytes / (n*sizeof(value_type)), actual_gb_s, actual_gb_s / BCL::nprocs());

	BCL::finalize();

	return 0;
}
