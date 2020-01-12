SHELL='bash'

# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../../../

BACKEND = $(shell echo $(BCL_BACKEND) | tr '[:lower:]' '[:upper:]')

TIMER_CMD=time

ifeq ($(BACKEND),SHMEM)
  BACKEND=SHMEM
  BCLFLAGS = -DSHMEM -I$(BCLROOT)
  CXX=CC

  BCL_RUN=oshrun -n 4
else ifeq ($(BACKEND),GASNET_EX)
  BACKEND=GASNET_EX
  # XXX: Allow selection of conduit.
  include $(gasnet_prefix)/include/aries-conduit/aries-par.mak

  BCLFLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DGASNET_EX -I$(BCLROOT)
  CXX = CC

  BCL_RUN=mpirun -n 4
else
  BACKEND=MPI
  BCLFLAGS = -I$(BCLROOT)
  CXX=CC

  BCL_RUN=mpirun -n 4
endif

FUTAR_FLAGS = -I/global/u1/b/brock/src/research/bcl/examples/benchmarks/mer-bench/experimental/futar

CXXFLAGS = -O3 -std=gnu++17 $(BCLFLAGS) $(FUTAR_FLAGS)

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	@echo "C $@ $(BACKEND)"
	@time $(CXX) -o $@ $^ $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

test: all
	@for target in $(TARGETS) ; do \
		echo "R $$target $(BACKEND)" ;\
	  time $(BCL_RUN) ./$$target || (echo "$$target $(BACKEND) FAIL $$?"; exit 1) ;\
	done

clean:
	@rm -f $(TARGETS)
