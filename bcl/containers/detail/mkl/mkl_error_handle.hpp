#pragma once

#include <mkl.h>
#include <string>

namespace BCL {

void mkl_error_handle(sparse_status_t status, std::string lbl = "") {
  std::string prefix = "BCL: Internal MKL error: ";
  if (status == SPARSE_STATUS_SUCCESS) {
  } else if (status == SPARSE_STATUS_NOT_INITIALIZED) {
    throw std::runtime_error(prefix + "SPARSE_STATUS_NOT_INITIALIZED: [" + lbl + "]");
  } else if (status == SPARSE_STATUS_ALLOC_FAILED) {
    throw std::runtime_error(prefix + "SPARSE_STATUS_ALLOC_FAILED: [" + lbl + "]");
  } else if (status == SPARSE_STATUS_INVALID_VALUE) {
    throw std::runtime_error(prefix + "SPARSE_STATUS_INVALID_VALUE: [" + lbl + "]");
  } else if (status == SPARSE_STATUS_EXECUTION_FAILED) {
    throw std::runtime_error(prefix + "SPARSE_STATUS_EXECUTION_FAILED: [" + lbl + "]");
  } else if (status == SPARSE_STATUS_INTERNAL_ERROR) {
    throw std::runtime_error(prefix + "SPARSE_STATUS_INTERNAL_ERROR: [" + lbl + "]");
  } else if (status == SPARSE_STATUS_NOT_SUPPORTED) {
    throw std::runtime_error(prefix + "SPARSE_STATUS_NOT_SUPPORTED: [" + lbl + "]");
  } else {
    throw std::runtime_error(prefix + "Unrecognized MKL sparse_status_t");
  }
}

}
