SHELL='bash'

CXX = CC

# XXX: Allow selection of conduit.
include $(gasnet_prefix)/include/aries-conduit/aries-par.mak

GASNET_FLAGS = -lpthread $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DGASNET_EX -I$(BCLROOT)

CXXFLAGS = -O3 -std=gnu++17 $(BCLFLAGS)

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(GASNET_FLAGS)

clean:
	@rm -f $(TARGETS)
