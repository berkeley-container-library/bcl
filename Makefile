#MPI_HOME=/home/agoswami/openmpi/openmpi-3.1.0/install
#CUDA_HOME=/cm/extra/apps/CUDA/9.0.176_384.81
#SHMEM_HOME=/home/yuxinc/nvshmem_0.1.0+cuda9_x86_64
#CUDA_HOME=/cm/extra/apps/CUDA.linux86-64/9.2.88_396.26
#SHMEM_HOME=/home/yuxinc/yuxinchenPSG_Home/nvshmem_0.1.0+cuda9_x86_64
BCL_HOME=/home/yuxinc/yuxinchenPSG_Home/bcl/bcl

CC=g++
CUDACC=${CUDA_HOME}/bin/nvcc

CUDACFLAGS=-O3 -DNVSHMEM_TARGET -c -dc --std=c++11 -gencode arch=compute_70,code=sm_70 -Xptxas="-v" -lineinfo --expt-extended-lambda -Xcompiler -fopenmp -I${SHMEM_HOME}/include/nvprefix -I${CUB_HOME}
#LDFLAGS =-gencode=arch=compute_70,code=sm_70 -L$(SHMEM_HOME)/lib -lshmem -lcuda
LDFLAGS =-arch=sm_70 -L$(SHMEM_HOME)/lib/nvprefix -lnvshmem -lcuda -lgomp 

OBJ=main.o

all: ${OBJ}
	${CUDACC} -o test ${OBJ} ${LDFLAGS}

%.o: %.cu
	${CUDACC} ${CUDACFLAGS} $<
clean:
	rm -rf *.o test
