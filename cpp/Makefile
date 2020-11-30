CXX=g++
CFLAGS=-Wall -std=c++11
PYTHON_DIR=-I/Users/egor/anaconda3/include/python3.7m -I/Users/egor/anaconda3/lib/python3.7/site-packages/pybind11/include
SOURCES=cpp_numpy.cpp
OBJECTS=$(SOURCES:.cpp=.o)
PROGRAM=np_fft$(shell python3-config --extension-suffix)

all: $(PROGRAM)

$(PROGRAM) : $(SOURCES)
	$(CXX) $(SOURCES) -undefined dynamic_lookup $(PYTHON_DIR) $(CFLAGS) -O3 -o $@ -shared

