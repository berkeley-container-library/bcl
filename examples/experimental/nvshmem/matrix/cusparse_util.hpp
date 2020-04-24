
#pragma once

#include <cusparse.h>

namespace BCL {

namespace cuda {

void throw_cusparse(cusparseStatus_t status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    if (status == CUSPARSE_STATUS_INTERNAL_ERROR) {
      throw std::runtime_error("CUSPARSE_STATUS_INTERNAL_ERROR");
    } else if (status == CUSPARSE_STATUS_INVALID_VALUE) {
      throw std::runtime_error("CUSPARSE_STATUS_INVALID_VALUE");
    } else if (status == CUSPARSE_STATUS_ALLOC_FAILED) {
      throw std::runtime_error("CUSPARSE_STATUS_ALLOC_FAILED");
    } else if (status == CUSPARSE_STATUS_NOT_INITIALIZED) {
      throw std::runtime_error("CUSPARSE_STATUS_NOT_INITIALIZED");
    } else if (status == CUSPARSE_STATUS_ARCH_MISMATCH) {
      throw std::runtime_error("CUSPARSE_STATUS_ARCH_MISMATCH");
    } else if (status == CUSPARSE_STATUS_EXECUTION_FAILED) {
      throw std::runtime_error("CUSPARSE_STATUS_EXECUTION_FAILED");
    } else if (status == CUSPARSE_STATUS_INTERNAL_ERROR) {
      throw std::runtime_error("CUSPARSE_STATUS_INTERNAL_ERROR");
    } else if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
      throw std::runtime_error("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");
    } else if (status == CUSPARSE_STATUS_NOT_SUPPORTED) {
      throw std::runtime_error("CUSPARSE_STATUS_NOT_SUPPORTED");
    }
    throw std::runtime_error("Unknown CUSPARSE error");
  }
}

} // end cuda

} // end BCL