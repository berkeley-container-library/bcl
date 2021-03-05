
# Makefile for some simple examples in this directory.
# On bridges, you need to run `source prep.sh` to set
# your environment before invoking this Makefile.

BCL_HOME=$(PWD)/../../../../

SOURCES += $(wildcard *.cu)
TARGETS := $(patsubst %.cu, %, $(SOURCES))

CXX=nvcc

# NVSHMEM_FLAGS=-DNVSHMEM_TARGET -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -ccbin g++  -I$(CUDA_HOME)/include/nvprefix -I$(NVSHMEM_HOME)/include/nvprefix -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT -rdc=true  -L$(NVSHMEM_HOME)/lib/nvprefix -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib -lmpi -lopen-rte -lopen-pal -lm -lnuma -ldl -lrt -lutil
NVSHMEM_FLAGS=-DNVSHMEM_TARGET -arch=sm_70 -ccbin g++  -I$(CUDA_HOME)/include -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT -rdc=true  -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib -lmpi_ibm -lopen-rte -lopen-pal -lm -lnuma -ldl -lrt -lutil

CXXFLAGS = -std=c++14 -O3 -I$(BCL_HOME) --expt-extended-lambda $(NVSHMEM_FLAGS)

all: $(TARGETS)

%: %.cu
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -fv $(TARGETS)
