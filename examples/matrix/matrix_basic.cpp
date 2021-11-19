// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <bcl/containers/SPMatrix.hpp>

float multiply_by_two(float x) {
  return 2 * x;
}

int main(int argc, char** argv) {
  BCL::init();

  BCL::DMatrix<float> a({8, 8});
  BCL::DMatrix<float> b({8, 8});

  a = 1;

  BCL::print("Initial matrix contents:\n");
  a.print();

  a.apply_inplace([](float x) { return x*23; });

  BCL::print("After multiplying by two:\n");
  a.print();

  a.apply_inplace([](float x) { return x*23; });

  BCL::finalize();
  return 0;
}