CXX=CC

# GASNET_FLAGS = -D_GNU_SOURCE=1 -DGASNET_PAR -D_REENTRANT    -I/usr/common/ftg/upcxx/snapshot-2018-07-20/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.opt/include -I/usr/common/ftg/upcxx/snapshot-2018-07-20/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.opt/include/aries-conduit -DGASNET_CONDUIT_ARIES  -O2   -Wno-format -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address    -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000  -Wno-format -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address      -L/usr/common/ftg/upcxx/snapshot-2018-07-20/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.opt/lib -lgasnet-aries-par -L/opt/cray/pe/pmi/5.0.13/lib64 -L/opt/cray/ugni/6.0.14-6.0.5.0_16.9__g19583bb.ari/lib64 -lpmi -lugni -L/opt/cray/udreg/2.3.2-6.0.5.0_13.12__ga14955a.ari/lib64 -ludreg  -lpthread -L/opt/cray/xpmem/2.2.4-6.0.5.1_8.18__g35d5e73.ari/lib64 -lxpmem -lhugetlbfs -L/opt/gcc/7.3.0/snos/lib/gcc/x86_64-suse-linux/7.3.0 -lgcc -lm

GASNET_FLAGS = -D_GNU_SOURCE=1 -DGASNET_PAR -D_REENTRANT    -I/usr/common/ftg/upcxx/2018.9.0/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.opt/include -I/usr/common/ftg/upcxx/2018.9.0/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.opt/include/aries-conduit -DGASNET_CONDUIT_ARIES  -O2   -Wno-format -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address    -DGASNET_EX -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000  -Wno-format -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address      -L/usr/common/ftg/upcxx/2018.9.0/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.opt/lib -lgasnet-aries-par -L/opt/cray/pe/pmi/5.0.13/lib64 -L/opt/cray/ugni/6.0.14-6.0.5.0_16.9__g19583bb.ari/lib64 -lpmi -lugni -L/opt/cray/udreg/2.3.2-6.0.5.0_13.12__ga14955a.ari/lib64 -ludreg  -lpthread -L/opt/cray/xpmem/2.2.4-6.0.5.1_8.18__g35d5e73.ari/lib64 -lxpmem -lhugetlbfs -L/opt/gcc/7.3.0/snos/lib/gcc/x86_64-suse-linux/7.3.0 -lgcc -lm

GASNET_DEBUG_FLAGS=-D_GNU_SOURCE=1 -DGASNET_PAR -D_REENTRANT    -I/usr/common/ftg/upcxx/2018.9.0/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.debug/include -I/usr/common/ftg/upcxx/2018.9.0/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.debug/include/aries-conduit -DGASNET_CONDUIT_ARIES  -g3   -Wno-format -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address    -DGASNET_EX -g3  -Wno-format -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address      -L/usr/common/ftg/upcxx/2018.9.0/hsw/gnu/PrgEnv-gnu-6.0.4-7.3.0/gasnet.debug/lib -lgasnet-aries-par -L/opt/cray/pe/pmi/5.0.13/lib64 -L/opt/cray/ugni/6.0.14-6.0.5.0_16.9__g19583bb.ari/lib64 -lpmi -lugni -L/opt/cray/udreg/2.3.2-6.0.5.0_13.12__ga14955a.ari/lib64 -ludreg  -lpthread -L/opt/cray/xpmem/2.2.4-6.0.5.1_8.18__g35d5e73.ari/lib64 -lxpmem -lhugetlbfs -L/opt/gcc/7.3.0/snos/lib/gcc/x86_64-suse-linux/7.3.0 -lgcc -lm

FUTAR_FLAGS = -I $(HOME)/src/tmp/other/thread-runtime

CPPFLAGS = -std=gnu++17 -O3
# CPPFLAGS = -std=gnu++17 -g

SRC = KmerIterator.cpp \
			hash.cpp \
			KmerMiddle.cpp \
			Kmer.cpp

OBJ=$(SRC:.cpp=.o)

all: simplified_kmer

clean:
	rm -fv *.o simplified_kmer

simplified_kmer: $(OBJ) simplified_kmer.cpp
	$(CXX) $(OBJ) simplified_kmer.cpp $(CPPFLAGS) -o $@ $(FUTAR_FLAGS) $(GASNET_FLAGS)

.cpp.o:
		$(CXX) $< -c -o $@ $(INCLUDE) $(CPPFLAGS)
