SHELL='bash'

# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../../

BACKEND = $(shell echo $(BCL_BACKEND) | tr '[:lower:]' '[:upper:]')

TIMER_CMD=time

ifeq ($(BACKEND),SHMEM)
  BACKEND=SHMEM
  BCLFLAGS = -DBCL_BACKEND_SHMEM -I$(BCLROOT)
  CXX = CC
  # CXX=oshc++

  BCL_RUN=oshrun -n 4
else ifeq ($(BACKEND),GASNET_EX)
  BACKEND=GASNET_EX
  # XXX: Allow selection of conduit.
  include $(gasnet_prefix)/include/mpi-conduit/mpi-par.mak

  BCLFLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DBCL_BACKEND_GASNET_EX -I$(BCLROOT)
  CXX = $(GASNET_CXX)

  BCL_RUN=mpirun -n 4
else
  BACKEND=MPI
  BCLFLAGS = -I$(BCLROOT) -DBCL_BACKEND_MPI
  CXX=CC

  BCL_RUN=mpirun -n 4
endif

COMPILER = $(shell CC --version)
IS_ICC = $(findstring icc, $(COMPILER))

ifneq (,$(IS_ICC))
  MKLFLAGS = -qopenmp $(COMBBLAS_FLAGS) -I $(HOME)/src/pkg/mtspgemmlib -I $(HOME)/src/pkg/mtspgemmlib/mtspgemmlib/GTgraph/sprng2.0-lite/include  -DMKL_ILP64 -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -dynamic -ltbb -ltbbmalloc
else
  MKLFLAGS = -fopenmp $(COMBBLAS_FLAGS) -I $(HOME)/src/pkg/mtspgemmlib -I $(HOME)/src/pkg/mtspgemmlib/mtspgemmlib/GTgraph/sprng2.0-lite/include  -DMKL_ILP64 -m64 -I${MKLROOT}/include  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl -dynamic -ltbb -ltbbmalloc
endif

CXXFLAGS = -O3 -std=gnu++17 $(BCLFLAGS) $(MKLFLAGS)

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	echo "C $@ $(BACKEND)"
	time $(CXX) -o $@ $^ $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

test: all
	@for target in $(TARGETS) ; do \
		echo "R $$target $(BACKEND)" ;\
	  time $(BCL_RUN) ./$$target || (echo "$$target $(BACKEND) FAIL $$?"; exit 1) ;\
	done

clean:
	@rm -f $(TARGETS)
