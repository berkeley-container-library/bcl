#pragma once

#include <bcl/bcl.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace shm {

void* open_shared_segment(const std::string& shm_key, size_t size) {
  size_t segment_size = size;

  int  fd = shm_open(shm_key.c_str(), O_CREAT | O_RDWR, 0777);
  if (fd == -1) {
    throw std::runtime_error("Failed to open shared memory object.");
  }

  if (BCL::rank() == 0) {
    int rv = ftruncate(fd, segment_size);
    if (rv == -1) {
      throw std::runtime_error("Could not up shared memory object to size " + std::to_string(size));
    }
  }

  BCL::barrier();

  void* ptr = mmap(nullptr, segment_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);

  if (ptr == MAP_FAILED) {
    throw std::runtime_error("Failed to mmap shared memory object.");
  }

  return ptr;
}

void close_shared_segment(void* ptr, const std::string& shm_key, size_t size) {
  fprintf(stderr, "munmap'ing...\n");
  munmap(ptr, size);
  fprintf(stderr, "done! (%lu)\n", BCL::rank());
  BCL::barrier();

  if (BCL::rank() == 0) {
    fprintf(stderr, "Unlinking...\n");
    shm_unlink(shm_key.c_str());
    fprintf(stderr, "Donw with unlinking...\n");
  }
}

} // end shm
