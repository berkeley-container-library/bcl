#pragma once

#include <map>
#include <unordered_map>
#include <limits>

namespace BCL {

double row_comm = 0;

template <typename T, typename MatrixType>
struct row_cache {
	row_cache(size_t size_bytes, size_t row_size, const MatrixType& matrix)
	  : matrix_(matrix) {
    capacity_ = std::max<size_t>(size_bytes / (row_size*sizeof(T)), 1);
	}

	std::vector<T>& get_row(size_t idx) {
		auto iter = rows_.find(idx);

		if (iter != rows_.end()) {
			hits_++;
			// fprintf(stderr, "(%lu) Reading cached copy of row %lu (cache holding %lu)\n", BCL::rank(), idx, rows_.size());
			// Remove entry in `last_used_`, since we
			// will be updating it with the current clock.
			last_used_.erase(iter->second.second);
			iter->second.second = clock_;
		} else {
			misses_++;
			// fprintf(stderr, "(%lu) Cache miss on row %lu\n", BCL::rank(), idx);
			auto begin = std::chrono::high_resolution_clock::now();
			rows_[idx] = {matrix_.get_row(idx), clock_};
			auto end = std::chrono::high_resolution_clock::now();
			double duration = std::chrono::duration<double>(end - begin).count();
			row_comm += duration;
			if (rows_.size() > capacity_) {
				// fprintf(stderr, "(%lu) %lu rows > %lu, deleting\n", BCL::rank(), rows_.size(), capacity_);
			  delete_oldest();
		  }
		}
		last_used_.insert({clock_, idx});
		clock_++;
		return rows_[idx].first;
	}

	void delete_oldest() {
		auto iter = last_used_.begin();

    if (iter != last_used_.end()) {
    	evictions_++;
    	// fprintf(stderr, "(%lu) deleting...\n", BCL::rank());
			size_t idx = iter->second;
			last_used_.erase(iter);

			rows_.erase(idx);
	  } else {
	  	// fprintf(stderr, "(%lu) nothing to delete...\n", BCL::rank());
	  }
	}

	void print_stats() {
		printf("(%lu) Hits: %lu. Misses: %lu. Evictions: %lu.  Hit rate (%lf%%)\n",
			     BCL::rank(), hits_, misses_, evictions_, 100.0*(double(hits_) / (hits_ + misses_)));
	}

	// Map from row idx -> [row, clock]
	std::unordered_map<size_t, std::pair<std::vector<T>, size_t>> rows_;
	// Map from clock -> row idx
	std::map<size_t, size_t> last_used_;
  // capacity in rows
	size_t capacity_;
	size_t clock_ = 0;

	size_t hits_ = 0;
	size_t misses_ = 0;
	size_t evictions_ = 0;

	MatrixType& matrix_;
};

template <typename T, typename I>
void rowwise_gemm(const BCL::SPMatrix<T, I>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {

	// Because of local c opt.
	assert(c.grid_shape()[1] == 1);

	for (size_t i = 0; i < a.grid_shape()[0]; i++) {
		for (size_t k = 0; k < a.grid_shape()[1]; k++) {
			if (a.tile_locale(i, k) == BCL::rank()) {
				T* values = a.vals_[i*a.grid_shape()[1] + k].local();
				I* row_ptr = a.row_ptr_[i*a.grid_shape()[1] + k].local();
				I* col_ind = a.col_ind_[i*a.grid_shape()[1] + k].local();
				T* local_c = c.tile_ptr(i, 0).local();
				for (size_t i_ = 0; i_ < a.tile_shape(i, k)[0]; i_++) {
					for (size_t j_ptr = row_ptr[i_]; j_ptr < row_ptr[i_+1]; j_ptr++) {
						size_t k_ = col_ind[j_ptr];

						size_t i__ = i_ + i*a.tile_shape()[0];
						size_t k__ = k_ + k*a.tile_shape()[1];

						auto value = values[j_ptr];

            auto begin = std::chrono::high_resolution_clock::now();
						auto row = b.get_row(k__);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end - begin).count();
            row_comm += duration;

						for (size_t j_ = 0; j_ < row.size(); j_++) {
							// c(i__, j_) = c(i__, j_) + row[j_]*value;
							local_c[i_*c.tile_shape()[1] + j_] += row[j_]*value;
						}
					}
				}
			}
		}
	}
}

template <typename T, typename I>
void cached_rowwise_gemm(const BCL::SPMatrix<T, I>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c, size_t cache_size) {
  // BCL::print("Cache of size %lu\n", cache_size);
	row_cache<T, const BCL::DMatrix<T>> cache(cache_size, b.shape()[1], b);

	// Because of local c opt.
	assert(c.grid_shape()[1] == 1);

	for (size_t i = 0; i < a.grid_shape()[0]; i++) {
		for (size_t k = 0; k < a.grid_shape()[1]; k++) {
			if (a.tile_locale(i, k) == BCL::rank()) {
				T* values = a.vals_[i*a.grid_shape()[1] + k].local();
				I* row_ptr = a.row_ptr_[i*a.grid_shape()[1] + k].local();
				I* col_ind = a.col_ind_[i*a.grid_shape()[1] + k].local();
				T* local_c = c.tile_ptr(i, 0).local();
				for (size_t i_ = 0; i_ < a.tile_shape(i, k)[0]; i_++) {
					for (size_t j_ptr = row_ptr[i_]; j_ptr < row_ptr[i_+1]; j_ptr++) {
						size_t k_ = col_ind[j_ptr];

						size_t i__ = i_ + i*a.tile_shape()[0];
						size_t k__ = k_ + k*a.tile_shape()[1];

						auto value = values[j_ptr];

						auto& row = cache.get_row(k__);
						// auto row = b.get_row(k__);

						for (size_t j_ = 0; j_ < row.size(); j_++) {
							// c(i__, j_) = c(i__, j_) + row[j_]*value;
							local_c[i_*c.tile_shape()[1] + j_] += row[j_]*value;
						}

					}
				}
			}
		}
	}
	cache.print_stats();
}

template <typename T, typename I>
void batched_rowwise_gemm(const BCL::SPMatrix<T, I>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
	// Because of local c opt.
	assert(c.grid_shape()[1] == 1);

	for (size_t i = 0; i < a.grid_shape()[0]; i++) {
		for (size_t k = 0; k < a.grid_shape()[1]; k++) {
			if (a.tile_locale(i, k) == BCL::rank()) {
				std::vector<std::tuple<size_t, size_t, T>> indices;
				T* values = a.vals_[i*a.grid_shape()[1] + k].local();
				I* row_ptr = a.row_ptr_[i*a.grid_shape()[1] + k].local();
				I* col_ind = a.col_ind_[i*a.grid_shape()[1] + k].local();
				T* local_c = c.tile_ptr(i, 0).local();
				for (size_t i_ = 0; i_ < a.tile_shape(i, k)[0]; i_++) {
					for (size_t j_ptr = row_ptr[i_]; j_ptr < row_ptr[i_+1]; j_ptr++) {
						size_t k_ = col_ind[j_ptr];

						size_t i__ = i_ + i*a.tile_shape()[0];
						size_t k__ = k_ + k*a.tile_shape()[1];

						auto value = values[j_ptr];

						indices.push_back({i_, k__, value});
					}
				}

				std::sort(indices.begin(), indices.end(), [](auto a, auto b) {
					return std::get<1>(a) < std::get<1>(b);
				});

        std::vector<T> row;
        size_t current_row = std::numeric_limits<size_t>::max();
				for (const auto& v : indices) {
					auto&& [i_, k_, value] = v;

					if (k_ != current_row) {
						auto begin = std::chrono::high_resolution_clock::now();
						row = b.get_row(k_);
						auto end = std::chrono::high_resolution_clock::now();
						double duration = std::chrono::duration<double>(end - begin).count();
						row_comm += duration;
						current_row = k_;
					}

					for (size_t j_ = 0; j_ < row.size(); j_++) {
						local_c[i_*c.tile_shape()[1] + j_] += row[j_]*value;
					}
				}
			}
		}
	}
}

} // end BCL