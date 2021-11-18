#pragma once

#include <cmath>

namespace BCL {

int zipf_dist(double alpha, int M)
{
  static int init_done = 0;
  static double k = 0;
  static double *sum_probs;
  static int prev_M = 0;
  double z;
  int value;
  int    i;
  int low, high, mid;

  if (prev_M != M) {
    init_done = 0;
    prev_M = M;
  }

  if (!init_done) {
    for (i=1; i<=M; i++)
      k = k + (1.0 / pow((double) i, alpha));
    k = 1.0 / k;

    sum_probs = (double *) malloc((M+1)*sizeof(double));
    sum_probs[0] = 0;
    for (i=1; i<=M; i++) {
      sum_probs[i] = sum_probs[i-1] + k / pow((double) i, alpha);
    }
    init_done = 1;
  }

  do {
    z = drand48();
  } while ((z == 0) || (z == 1));

  low = 1, high = M, mid;
  do {
    mid = floor((low+high)/2);
    if (sum_probs[mid] >= z && sum_probs[mid-1] < z) {
      value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid-1;
    } else {
      low = mid+1;
    }
  } while (low <= high);

  assert((value >=1) && (value <= M));

  return(value);
}

template <typename T, typename I, typename Blocking>
BCL::SPMatrix<T, I> generate_matrix(size_t m, size_t n, size_t nnz_per_row,
	                                  double alpha, Blocking&& block) {
	using value_type = T;
	using index_type = I;
  BCL::SPMatrix<value_type, index_type> matrix(m, n, std::move(block));

  for (size_t i = 0; i < matrix.grid_shape()[0]; i++) {
  	for (size_t j = 0; j < matrix.grid_shape()[1]; j++) {
  		if (matrix.tile_rank(i, j) == BCL::rank()) {

  			std::vector<value_type> values;
  			std::vector<index_type> rowptr(matrix.tile_shape(i, j)[0] + 1, 0);
  			std::vector<index_type> colind;

  			size_t nnz = 0;

  			for (size_t i_ = 0; i_ < matrix.tile_shape(i, j)[0]; i_++) {
  				size_t nnz_row = std::max<size_t>(1, drand48()*(nnz_per_row*2));

  				for (size_t nz_ = 0; nz_ < nnz_row; nz_++) {
  					size_t j_;

  					if (alpha == 0.0) {
  						j_ = drand48()*matrix.shape()[1];
  					} else {
  						j_ = zipf_dist(alpha, matrix.shape()[1]);
  					}

  					if (j_ >= matrix.tile_shape()[1]*j &&
  						  j_ < matrix.tile_shape()[1]*j + matrix.tile_shape(i, j)[1]) {
  						nnz++;
  					  colind.push_back(j_ - matrix.tile_shape()[1]*j);
  					}
  				}

  				rowptr[i_+1] = nnz;
  			}

  			values.resize(nnz, 1);

	  		BCL::CSRMatrix<value_type, index_type> local_mat(matrix.tile_shape(i, j)[0],
	  			                                               matrix.tile_shape(i, j)[1],
	  			                                               nnz,
	  			                                               std::move(values),
	  			                                               std::move(rowptr),
	  			                                               std::move(colind));

	  		matrix.assign_tile(i, j, local_mat);
  		}
  	}
  }

  BCL::barrier();

  matrix.rebroadcast_tiles();

  return matrix;
}

template <typename T, typename I, typename Blocking, typename TeamType>
BCL::SPMatrix<T, I> generate_matrix(size_t m, size_t n, size_t nnz_per_row,
                                    double alpha, Blocking&& block,
                                    TeamType&& team) {
  using value_type = T;
  using index_type = I;
  BCL::SPMatrix<value_type, index_type> matrix(m, n, std::move(block), std::forward<TeamType>(team));

  for (size_t i = 0; i < matrix.grid_shape()[0]; i++) {
    for (size_t j = 0; j < matrix.grid_shape()[1]; j++) {
      if (matrix.tile_rank(i, j) == BCL::rank()) {

        std::vector<value_type> values;
        std::vector<index_type> rowptr(matrix.tile_shape(i, j)[0] + 1, 0);
        std::vector<index_type> colind;

        size_t nnz = 0;

        for (size_t i_ = 0; i_ < matrix.tile_shape(i, j)[0]; i_++) {
          size_t nnz_row = std::max<size_t>(1, drand48()*(nnz_per_row*2));

          for (size_t nz_ = 0; nz_ < nnz_row; nz_++) {
            size_t j_;

            if (alpha == 0.0) {
              j_ = drand48()*matrix.shape()[1];
            } else {
              j_ = zipf_dist(alpha, matrix.shape()[1]);
            }

            if (j_ >= matrix.tile_shape()[1]*j &&
                j_ < matrix.tile_shape()[1]*j + matrix.tile_shape(i, j)[1]) {
              nnz++;
              colind.push_back(j_ - matrix.tile_shape()[1]*j);
            }
          }

          rowptr[i_+1] = nnz;
        }

        values.resize(nnz, 1);

        BCL::CSRMatrix<value_type, index_type> local_mat(matrix.tile_shape(i, j)[0],
                                                         matrix.tile_shape(i, j)[1],
                                                         nnz,
                                                         std::move(values),
                                                         std::move(rowptr),
                                                         std::move(colind));

        matrix.assign_tile(i, j, local_mat);
      }
    }
  }

  BCL::barrier();

  matrix.rebroadcast_tiles();

  return matrix;
}

} // end BCL