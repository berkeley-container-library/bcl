
# It may be a good idea to set a `BCL_HOME` global environment variable.
# That way this Makefile would work in any directory.
BCL_HOME ?= $(PWD)/../../

# Try to pick out whether `CC` exists (for NERSC systems)
# If not, try to use mpic++.
ifeq (, $(shell which CC))
	CXX = mpic++
else
	CXX = CC
endif

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

# SHMEM and GASNet-EX tend to perform the best.
# MPI is default
BACKEND = 
# BACKEND = '-DSHMEM'
# BACKEND = '-DGASNET_EX'

# If you compile off of Cori, you will need to link with `-lcblas`.
BLAS_LIB = 

CXXFLAGS = -std=gnu++17 -O3 $(BACKEND) -I$(BCL_HOME) $(BLAS_LIB)

all: $(TARGETS)

%: %.cu
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -fv $(TARGETS)
