#include <bcl/containers/sequential/CSRMatrix.hpp>

int main(int argc, char** argv) {
	size_t num_args = 1;

  if (argc != num_args+1) {
    fprintf(stderr, "usage: ./analyze_matrix [input file]\n");
    fprintf(stderr, "Analyze matrix.\n");
    fprintf(stderr, "Automatically infers type from extensions (.mtx or .binary).\n");
    return 1;
  }

  std::string input_fname(argv[1]);

  auto format = BCL::matrix_io::detect_file_type(input_fname);
  assert(format == BCL::FileFormat::Binary);

  using index_type = int;

  fprintf(stderr, "Reading in \"%s\"\n", input_fname.c_str());
  BCL::CSRMatrix<float, index_type> matrix(input_fname);

  fprintf(stderr, "Dimensions: %lu x %lu, NNZ: %lu\n", matrix.shape()[0], matrix.shape()[1], matrix.nnz());

  std::vector<size_t> nnzs(10*10, 0);

  size_t m_s = (matrix.shape()[0] + 10 - 1) / 10;
  size_t n_s = (matrix.shape()[1] + 10 - 1) / 10;

  for (size_t i = 0; i < matrix.m(); i++) {
    for (index_type j_ptr = matrix.rowptr_data()[i]; j_ptr < matrix.rowptr_data()[i+1]; j_ptr++) {
      index_type j = matrix.colind_data()[j_ptr];

      size_t tile_i = i / m_s;
      size_t tile_j = j / n_s;

      nnzs[tile_i*10 + tile_j]++;
    }
  }

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      // fprintf(stderr, "(%lu, %lu): %lu nnz\n", i, j, nnzs[i*10 + j]);
    }
  }

  size_t total = 0;
  size_t max = 0;
  for (const auto& nnz : nnzs) {
    total += nnz;
    max = std::max(max, nnz);
  }

  double avg = total / (10.0*10.0);
  double load_imbalance = max / avg;

  fprintf(stderr, "Avg NNZs %lf, max %lu, load imbalance %lf\n",
          avg, max, load_imbalance);

	return 0;
}
