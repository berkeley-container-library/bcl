SHELL='bash'

##################### BCL part

# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../../../

BACKEND = $(shell echo $(BCL_BACKEND) | tr '[:lower:]' '[:upper:]')
PLATFORM = $(shell echo $(BCL_PLATFORM) | tr '[:lower:]' '[:upper:]')

TIMER_CMD=time

ifeq ($(BACKEND),GASNET_EX)
  BACKEND=GASNET_EX
  # XXX: Allow selection of conduit.
  ifeq ($(PLATFORM), CORI)
    include $(gasnet_prefix)/include/aries-conduit/aries-par.mak
    CXX = CC
  else
    include $(gasnet_prefix)/include/mpi-conduit/mpi-par.mak
    CXX = mpic++
  endif

  BCLFLAGS = -lpthread $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DGASNET_EX -I$(BCLROOT)

  BCL_RUN=mpirun -n 4
else
  @echo "Only running ARH tests with GASNet_EX"
  exit 1
endif

CXXFLAGS = -std=gnu++17 $(BCLFLAGS)

##################### UPCXX part

CFLAGS = --std=c++11
# upcxx-meta PPFLAGS are really CFLAGS to be used during compilation
# upcxx-meta LDFLAGS are really CFLAGS to be used during linking
# upcxx-meta LIBFLAGS are really a comsbination of LDLIBS and LDFLAGS

UPCXXFLAGS = `upcxx-meta PPFLAGS` `upcxx-meta LDFLAGS`
LDFLAGS = `upcxx-meta LIBFLAGS`

############################################################3

TARGETS := kmer_hash_agg kmer_hash_upcxx_rpc kmer_hash_upcxx_rdma
all: $(TARGETS)

kmer_hash_naive: kmer_hash_naive.cpp kmer_t.hpp pkmer_t.hpp packing.hpp read_kmers.hpp
	@echo "C $@ $(BACKEND)"
	@time $(CXX) kmer_hash_naive.cpp -o kmer_hash_naive $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

kmer_hash_agg: kmer_hash_agg.cpp kmer_t.hpp pkmer_t.hpp packing.hpp read_kmers.hpp
	@echo "C $@ $(BACKEND)"
	@time $(CXX) kmer_hash_agg.cpp -o kmer_hash_agg $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

kmer_hash_upcxx_rpc: kmer_hash_upcxx_rpc.cpp kmer_t.hpp pkmer_t.hpp packing.hpp read_kmers.hpp hash_map_upcxx_rpc.hpp butil.hpp
	$(CXX) $(CFLAGS) kmer_hash_upcxx_rpc.cpp -o kmer_hash_upcxx_rpc $(UPCXXFLAGS) $(LDFLAGS)

kmer_hash_upcxx_rdma: kmer_hash_upcxx_rdma.cpp kmer_t.hpp pkmer_t.hpp packing.hpp read_kmers.hpp hash_map_upcxx_rdma.hpp butil.hpp
	$(CXX) $(CFLAGS) kmer_hash_upcxx_rdma.cpp -o kmer_hash_upcxx_rdma $(UPCXXFLAGS) $(LDFLAGS)

%: %.cpp kmer_t.hpp pkmer_t.hpp packing.hpp read_kmers.hpp
	@echo "C $@ $(BACKEND)"
	@time $(CXX) $< -o $@ $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

clear:
	@rm -fv *.dat my_solution.txt

clean:
	@rm -fv $(TARGETS) *.dat my_solution.txt