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

  BCL::SPMatrix<float> a("1138_bus.mtx");
  BCL::SPMatrix<float> b("1138_bus.mtx");

  auto c = a.dot(b);

  BCL::print("Printing out C matrix:\n");
  c.print();

  BCL::finalize();
  return 0;
}
