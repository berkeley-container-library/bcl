# NOTE: uncomment these three lines to use the GASNet-EX backend.
#       Note that the environment variable `gasnet_prefix`
#       must be set.
#       (You also need to set `BACKEND` to `-DBCL_BACKEND_GASNET_EX` below.
# include $(gasnet_prefix)/include/aries-conduit/aries-par.mak
# GASNET_FLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS)
# GASNET_BACKEND_LIB_FLAGS = $(GASNET_LDFLAGS) $(GASNET_LIBS)

# It may be a good idea to set a `BCL_HOME` global environment variable.
# That way this Makefile would work in any directory.
BCL_HOME ?= $(PWD)/../../

# Try to pick out whether `CC` exists (for NERSC systems)
# If not, try to use mpic++.
ifeq (, $(shell which CC))
	CXX = g++
else
	CXX = CC
endif

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

# SHMEM and GASNet-EX tend to perform the best.
# MPI is default
BACKEND = -DBCL_BACKEND_MPI
# BACKEND = -DBCL_BACKEND_SHMEM
# BACKEND = -DBCL_BACKEND_GASNET_EX $(GASNET_FLAGS)
LD_FLAGS = $(GASNET_BACKEND_LIB_FLAGS)

CXXFLAGS = -std=c++20 -DCPP20 -O3 $(BACKEND) -I$(BCL_HOME)

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) `mpic++ -showme:compile` `mpic++ -showme:link`

clean:
	rm -fv $(TARGETS)
