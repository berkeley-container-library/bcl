#pragma once

namespace BCL {

enum FileFormat {
  MatrixMarket,
  MatrixMarketZeroIndexed,
  Binary,
  Unknown
};

namespace matrix_io {

auto detect_file_type(const std::string& fname) {
  size_t suffix_spot = 0;

  for (int64_t i = fname.size()-1; i >= 0; i--) {
    if (fname[i] == '.') {
      suffix_spot = i;
    }
  }

  std::string suffix = fname.substr(suffix_spot);

  if (suffix == ".mtx") {
    return BCL::FileFormat::MatrixMarket;
  } else if (suffix == ".binary") {
    return BCL::FileFormat::Binary;
  } else {
    assert(false);
    return BCL::FileFormat::Unknown;
  }
}

struct matrix_shape {
  std::array<size_t, 2> shape;
  size_t nnz;
};

matrix_shape matrix_market_info(const std::string& fname, bool one_indexed = true);
matrix_shape binary_info(const std::string& fname);

matrix_shape matrix_info(const std::string& fname,
                         BCL::FileFormat format = BCL::FileFormat::Unknown) {
  if (format == FileFormat::Unknown) {
    format = detect_file_type(fname);
  }

  if (format == FileFormat::MatrixMarket) {
    return matrix_market_info(fname);
  } else if (format == FileFormat::MatrixMarketZeroIndexed) {
    return matrix_market_info(fname, false);
  } else if (format == FileFormat::Binary) {
    return binary_info(fname);
  } else {
    throw std::runtime_error("matrix_info: Could not detect file format for \""
                             + fname + "\"");
  }
}

matrix_shape matrix_market_info(const std::string& fname, bool one_indexed) {
  std::ifstream f;

  f.open(fname.c_str());

  if (!f.is_open()) {
    throw std::runtime_error("matrix_market_size: cannot open " + fname);
  }

  std::string buf;

  bool outOfComments = false;

  while (!outOfComments) {
    getline(f, buf);
    regex_t regex;
    int reti;

    reti = regcomp(&regex, "^%", 0);
    reti = regexec(&regex, buf.c_str(), 0, NULL, 0);

    if (reti == REG_NOMATCH) {
      outOfComments = true;
    }
  }

  size_t m, n, nnz;
  sscanf(buf.c_str(), "%lu %lu %lu", &m, &n, &nnz);

  f.close();

  return matrix_shape{{m, n}, nnz};
}

matrix_shape binary_info(const std::string& fname) {
  FILE* f = fopen(fname.c_str(), "r");
  size_t m, n, nnz;
  assert(f != NULL);
  size_t items_read = fread(&m, sizeof(size_t), 1, f);
  assert(items_read == 1);
  items_read = fread(&n, sizeof(size_t), 1, f);
  assert(items_read == 1);
  items_read = fread(&nnz, sizeof(size_t), 1, f);
  assert(items_read == 1);
  fclose(f);
  return matrix_shape{{m, n}, nnz};
}

} // end matrix_io

} // end BCL
