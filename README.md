# BCL
The Berkeley Container Library is a C++ library of distributed data structures
with interfaces similar to those in STL containers.

## Example

```cpp
  BCL::HashMap<std::string, int> map(1000);

  map[std::to_string(BCL::rank())] = BCL::rank();

  BCL::barrier();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      int value = *map.find(std::string(i), value);
      printf("Got key, val %s, %lu\n",
             std::to_string(i).c_str(), value);
    }
  }
```

## Design
### Global Pointers
BCL data structures are designed to operate using only one-sided operations
that can be executed with RDMA.  Under the hood, BCL data structures are
implemented using the BCL Core, which is a small API based around
[global pointers](https://people.eecs.berkeley.edu/~brock/blog/remote_pointers.php),
which are C++ objects that can be used like regular pointers, but
reference memory that may lie on another process or node.

```cpp
  BCL::GlobalPtr<int> ptr = nullptr;

  if (BCL::rank() == 0) {
    ptr = BCL::alloc<int>(BCL::nprocs());
  }

  ptr = BCL::broadcast(ptr, 0);

  ptr[BCL::rank()] = BCL::rank();

  if (BCL::rank() == 1) {
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      int local_val = local_ptr[i];
      printf("%lu: %lu\n", i, local_ptr[i]);
    }
  }
```
### Backends
The BCL Core works by calling a small set of backend functions that do
things like read or write from remote memory.  The interface is designed
to make it easy to implement new backend functions, and we have implemented
backends for MPI, OpenSHMEM, and GASNet-EX.

## Compiling BCL Programs
BCL is a header-only library, so to compile a program with BCL, you only need to
include the necessary BCL header files and compile as normal for your backend.

```bash
[xiii@shini example-dir]$ vi hello.cpp
[xiii@shini example-dir]$ mpic++ hello.cpp -o hello -std=gnu++17 -O3 -I $HOME/src/BCL
[xiii@shini example-dir]$ mpirun -n 4 ./hello
Hello, BCL! I am rank 1/4.
Hello, BCL! I am rank 0/4.
Hello, BCL! I am rank 2/4.
Hello, BCL! I am rank 3/4.
```

MPI is the default backend, and you can explicitly select a backend with the compiler
directives `-DMPI`, `-DSHMEM`, and `-DGASNET_EX`.  See `/bcl/bcl.hpp` for details.

## Example Programs
You can find example BCL programs in the /examples directory.  We recommend you start
with /examples/simple, which covers basic functions in the BCL Core.

## Bugs
Please let us know if you identify any bugs or general usability issues by creating
an issue on GitHub or directly contacting the authors.

## Academic Papers

[[ICPP'19](https://arxiv.org/abs/1810.13029)] "BCL: A Cross-Platform Distributed Container Library"

## Blog Posts

[[1](https://people.eecs.berkeley.edu/~brock/blog/remote_pointers.php)] "Creating Expressive C++ Smart Pointers for Remote Memory"

[[2](https://people.eecs.berkeley.edu/~brock/blog/storing_cpp_objects.php)] "Storing C++ Objects in Distributed Memory"
