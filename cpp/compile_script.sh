c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` cpp_numpy.cpp -o example`python3-config --extension-suffix`
