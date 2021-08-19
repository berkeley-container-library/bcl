#pragma once

namespace BCL {

template <typename T, typename I>
void rowwise_gemm(const BCL::SPMatrix<T, I>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
	for (size_t i = 0; i < a.grid_shape()[0]; i++) {
		for (size_t k = 0; k < a.grid_shape()[1]; k++) {
			if (a.tile_locale(i, k) == BCL::rank()) {
				T* values = a.vals_[i*a.grid_shape()[0] + k].local();
				I* row_ptr = a.row_ptr_[i*a.grid_shape()[0] + k].local();
				I* col_ind = a.col_ind_[i*a.grid_shape()[0] + k].local();
				for (size_t i_ = 0; i_ < a.tile_shape(i, k)[0]; i_++) {
					for (size_t j_ptr = row_ptr[i_]; j_ptr < row_ptr[i_+1]; j_ptr++) {
						size_t k_ = col_ind[j_ptr];
						auto value = values[j_ptr];

						auto row = b.get_row(k_);

						for (size_t j_ = 0; j_ < row.size(); j_++) {
							c(i_, j_) = c(i_, j_) + row[j_]*value;
						}

					}
				}
			}
		}
	}
}

} // end BCL