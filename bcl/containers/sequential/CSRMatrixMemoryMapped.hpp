#pragma once

#include <vector>
#include <bcl/containers/sequential/CSRMatrix.hpp>

#include <stdexcept>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace BCL {

class MemoryMappedFile {
public:
  MemoryMappedFile(const std::string& fname) : fname_(fname) {
    fd_ = open(fname.c_str(), O_RDONLY);

    if (fd_ < 0) {
      throw std::runtime_error("MemoryMappedFile::MemoryMappedFile(...): could not open file \"" + fname + "\"");
    }

    struct stat st;
    stat(fname_.c_str(), &st);
    length_ = st.st_size;

    address_ = mmap(NULL, length_, PROT_READ, MAP_SHARED, fd_, 0);
  }

  const void* data() {
    return address_;
  }

  size_t size() const {
    return length_;
  }

  ~MemoryMappedFile() {
    munmap(address_, length_);
    close(fd_);
  }

private:
  std::string fname_;
  int fd_;

  void* address_;
  size_t length_;
};

template <typename T, typename I>
class CSRMatrixMemoryMapped {
public:
  using value_type = T;
  using index_type = I;
  using size_type = size_t;

  CSRMatrixMemoryMapped(const std::string& fname) : file_(fname) {
    const size_t& m = *((const size_t *) file_.data());
    const size_t& n = *(((const size_t *) file_.data()) + 1);
    const size_t& nnz = *(((const size_t *) file_.data()) + 2);

    const T* values = (const T*) (((const size_t *) file_.data()) + 3);
    const index_type* colind = (const index_type*) (values + nnz);
    const index_type* rowptr = colind + nnz;

    m_ = m;
    n_ = n;
    nnz_ = nnz;

    values_ = values;
    colind_ = colind;
    rowptr_ = rowptr;
  }

  const T* values_data() const {
    return values_;
  }

  const index_type* colind_data() const {
    return colind_;
  }

  const index_type* rowptr_data() const {
    return rowptr_;
  }

  size_t m() const {
    return m_;
  }

  size_t n() const {
    return n_;
  }

  size_t nnz() const {
    return nnz_;
  }
  
  template <typename Allocator = std::allocator<T>>
  CSRMatrix<T, index_type> get_slice_impl_(size_type imin, size_type imax,
  	                                       size_type jmin,
                                           size_type jmax) {
    using rebind_type = typename Allocator:: template rebind<index_type>;
    using IAllocator = typename rebind_type::other;
    std::vector<T, Allocator> vals;
    std::vector<index_type, IAllocator> row_ptr;
    std::vector<index_type, IAllocator> col_ind;

    imin = std::max(imin, size_type(0));
    imax = std::min(imax, m_);
    jmax = std::min(jmax, n_);
    jmin = std::max(jmin, size_type(0));

    size_type m = imax - imin;
    size_type n = jmax - jmin;

    row_ptr.resize(m+1);

    assert(imin <= imax && jmin <= jmax);

    // TODO: there's an early exit possible when
    //       column indices are sorted.

    size_type new_i = 0;
    for (size_type i = imin; i < imax; i++) {
      row_ptr[i - imin] = new_i;
      for (index_type j = rowptr_[i]; j < rowptr_[i+1]; j++) {
        if (colind_[j] >= jmin && colind_[j] < jmax) {
          vals.push_back(values_[j]);
          col_ind.push_back(colind_[j] - jmin);
          new_i++;
        }
      }
    }

    size_type nnz = vals.size();
    row_ptr[m] = nnz;

    return CSRMatrix<T, index_type>(m, n, nnz,
    	                              std::move(vals),
    	                              std::move(row_ptr),
                                    std::move(col_ind));
  }

  MemoryMappedFile file_;
  size_t m_;
  size_t n_;
  size_t nnz_;
  const T* values_;
  const index_type* colind_;
  const index_type* rowptr_;
};

class MemoryMappedFileWritable {
public:
  MemoryMappedFileWritable(const std::string& fname, size_t size = 0) : fname_(fname) {
    fd_ = open(fname.c_str(), O_CREAT | O_RDWR);

    if (fd_ < 0) {
      throw std::runtime_error("MemoryMappedFileWriteable::MemoryMappedFileWriteable(...): could not open file \"" + fname + "\"");
    }

    struct stat st;
    stat(fname_.c_str(), &st);
    length_ = st.st_size;

    if (size != 0) {
      length_ = size;
      ftruncate(fd_, size);
    }

    address_ = mmap(NULL, length_, PROT_WRITE, MAP_SHARED, fd_, 0);
  }

  const void* data() {
    return address_;
  }

  size_t size() const {
    return length_;
  }

  ~MemoryMappedFileWritable() {
    munmap(address_, length_);
    close(fd_);
  }

private:
  std::string fname_;
  int fd_;

  void* address_;
  size_t length_;
};

} // end BCL
