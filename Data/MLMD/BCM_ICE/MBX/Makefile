CXX=g++
CXXFLAGS= -Wall -std=c++11  -g -O2

MBX_HOME=/home/willow/CC/MBX-1.0.0
FFTW_HOME=/home/willow/mambaforge/
ifndef MBX_HOME
$(info "MBX_HOME is not set. Please set it or ignore if the default $(HOME)/codes/MBX is OK")
MBX_HOME = $(HOME)/codes/MBX
endif

LIBS = -l:libmbx.a -fopenmp -lfftw3
LIBDIR = -L$(MBX_HOME)/lib -L$(FFTW_HOME)/lib

INCLUDEDIR = -I$(MBX_HOME)/include -I$(FFTW_HOME)/include

BINDIR = ./

OBJDIR = ./

all: example example_pbc mbx_single_pbc.x

example: example.cpp $(BINDIR)/.sentinel
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(LIBDIR) example.cpp $(LIBS) -o $@

example_pbc: example_pbc.cpp $(BINDIR)/.sentinel
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(LIBDIR) example_pbc.cpp $(LIBS) -o $@

mbx_single_pbc.x: mbx_single_pbc.cpp $(BINDIR)/.sentinel
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $(LIBDIR) mbx_single_pbc.cpp $(LIBS) -o $@

$(BINDIR)/.sentinel:
	mkdir -p $(BINDIR)
	touch $@

clean:
	rm -rf example example_pbc
