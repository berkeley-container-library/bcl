# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../../../

# XXX: To compile with MPI
#BCLFLAGS = -I$(BCLROOT) -DBCL_BACKEND_MPI
#CXX=mpic++

# XXX: To compile with OpenSHMEM
BCLFLAGS = -DBCL_BACKEND_SHMEM -I$(BCLROOT)
# CXX=oshc++
CXX=CC

# XXX: Compiling with GASNet-EX varies
#      depending on conduit.  See your
#      GASNet documentation.
# BCLFLAGS = -DBCL_BACKEND_GASNET_EX -I$(BCLROOT)
# CXX=$(GASNET_CXX) ...

CXXFLAGS = -std=gnu++17 -O3 $(BCLFLAGS)

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -fv $(TARGETS)
