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
	size_t segment_size = 256;

	BCL::init(segment_size);
  BCL::SPMatrix<float, int> a("1138_bus.binary");

  size_t m = a.shape()[0];
  size_t k = a.shape()[1];

  size_t n = 1024;

	BCL::DMatrix<float> b({k, n});
	BCL::DMatrix<float> c({k, n});
	BCL::DMatrix<float> c_({k, n});

	b = 1;
	c = 0;
	c_ = 0;

	BCL::barrier();

	// BCL::rowwise_gemm(a, b, c);

	BCL::barrier();

	BCL::gemm(a, b, c_);

	BCL::finalize();

	return 0;
}
