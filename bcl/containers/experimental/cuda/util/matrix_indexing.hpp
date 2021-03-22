#pragma once

namespace BCL {

namespace cuda {

struct RowMajorIndexing {
	__host__ __device__
	size_t index(size_t i, size_t j, size_t ld) {
		return i*ld + j;
	}

	__host__ __device__
	size_t index(size_t i, size_t j, size_t m, size_t n) {
		return i*default_ld(m, n) + j;
	}

	__host__ __device__
	size_t default_ld(size_t m, size_t n) {
		return n;
	}

	__host__ __device__
	size_t size(size_t m, size_t n, size_t ld) {
		if (ld == 0) {
			ld = default_ld(m, n);
		}
		return m*ld;
	}
};

struct ColumnMajorIndexing {
	__host__ __device__
	size_t index(size_t i, size_t j, size_t ld) {
		return i + j*ld;
	}

	__host__ __device__
	size_t index(size_t i, size_t j, size_t m, size_t n) {
		return i + j*default_ld(m, n);
	}

	__host__ __device__
	size_t default_ld(size_t m, size_t n) {
		return m;
	}

	__host__ __device__
	size_t size(size_t m, size_t n, size_t ld) {
		if (ld == 0) {
			ld = default_ld(m, n);
		}
		return ld*n;
	}
};

} // end cuda
	
} // end BCL