// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

# NOTE: uncomment these three lines to use the GASNet-EX backend.
#       Note that the environment variable `gasnet_prefix`
#       must be set.
#       (You also need to set `BACKEND` to `-DGASNET_EX` below.
# include $(gasnet_prefix)/include/aries-conduit/aries-par.mak
# GASNET_FLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS)
# GASNET_BACKEND_LIB_FLAGS = $(GASNET_LDFLAGS) $(GASNET_LIBS)

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
# BACKEND = '-DGASNET_EX' $(GASNET_FLAGS)
LD_FLAGS = $(GASNET_BACKEND_LIB_FLAGS)

# If you compile off of Cori, you will need to link with `-lcblas`.
BLAS_LIB = 

CXXFLAGS = -std=gnu++17 -O3 $(BACKEND) -I$(BCL_HOME) $(BLAS_LIB)

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS)

clean:
	rm -fv $(TARGETS)
