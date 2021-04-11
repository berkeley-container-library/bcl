// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

# Makefile for some simple examples in this directory.
# On bridges, you need to run `source prep.sh` to set
# your environment before invoking this Makefile.

BCL_HOME=$(PWD)/../../../../

SOURCES += $(wildcard *.cu)
TARGETS := $(patsubst %.cu, %, $(SOURCES))

CXX=nvcc

# NVSHMEM_FLAGS=-DNVSHMEM_TARGET -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -ccbin g++  -I$(CUDA_HOME)/include/nvprefix -I$(NVSHMEM_HOME)/include/nvprefix -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT -rdc=true  -L$(NVSHMEM_HOME)/lib/nvprefix -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib -lmpi -lopen-rte -lopen-pal -lm -lnuma -ldl -lrt -lutil
GRAPHBLAST_DIR = /ccs/home/b2v/src/research/graphblast
GRAPHBLAST_FLAGS = -I$(GRAPHBLAST_DIR) -I$(GRAPHBLAST_DIR)/ext/moderngpu/include/ -I$(GRAPHBLAST_DIR)/ext/cub/cub/ $(GRAPHBLAST_DIR)/ext/moderngpu/src/mgpucontext.cu $(GRAPHBLAST_DIR)/ext/moderngpu/src/mgpuutil.cpp -lboost_program_options -lcublas -lcusparse -lcurand

MATRIX_SUM_FLAGS = -I/autofs/nccs-svm1_home2/b2v/src/research/matrix_sum
NSPARSE_FLAGS = -I/ccs/home/b2v/src/research/nsparse/cuda-cpp/inc -I$(CUDAPATH)/samples/common/inc

CUSP_FLAGS = -I/autofs/nccs-svm1_home2/b2v/pkg/cusplibrary

NVSHMEM_FLAGS=-DNVSHMEM_TARGET -arch=sm_35 -ccbin g++  -I$(CUDA_HOME)/include -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT -rdc=true  -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_HOME)/lib -lmpi_ibm -lopen-rte -lopen-pal -lm -lnuma -ldl -lrt -lutil -lcublas_static -lculibos -lcublasLt_static -I$(OLCF_ESSL_ROOT)/include -L$(OLCF_ESSL_ROOT)/lib64 -lessl

SHMEM_FLAGS=-lpthread -L/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/spectrum-mpi-10.3.1.2-20200121-chae23sgwacfeot7vxkpfboz6wao2c33/lib -loshmem -lmpi_ibm

CXXFLAGS = -std=c++14 -O3 -DSHMEM $(SHMEM_FLAGS) -I$(BCL_HOME) --expt-extended-lambda $(NVSHMEM_FLAGS) $(GRAPHBLAST_FLAGS) $(MATRIX_SUM_FLAGS) $(NSPARSE_FLAGS) $(CUSP_FLAGS)

all: $(TARGETS)

%: %.cu
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -fv $(TARGETS)
