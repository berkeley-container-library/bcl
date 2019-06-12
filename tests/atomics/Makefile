SHELL='bash'

# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../

BACKEND = $(shell echo $(BCL_BACKEND) | tr '[:lower:]' '[:upper:]')

TIMER_CMD=time

ifeq ($(BACKEND),SHMEM)
  BACKEND=SHMEM
  BCLFLAGS = -DSHMEM -I$(BCLROOT)
  CXX=oshc++

  BCL_RUN=oshrun -n 4
else ifeq ($(BACKEND),GASNET_EX)
  BACKEND=GASNET_EX
  # XXX: Allow selection of conduit.
  include $(gasnet_prefix)/include/mpi-conduit/mpi-par.mak

  BCLFLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DGASNET_EX -I$(BCLROOT)
  CXX = mpic++

  BCL_RUN=mpirun -n 4
else
  BACKEND=MPI
  BCLFLAGS = -I$(BCLROOT)
  CXX=mpic++

  BCL_RUN=mpirun -n 4
endif

CXXFLAGS = -std=gnu++17 $(BCLFLAGS)

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
