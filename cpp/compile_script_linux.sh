g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_numpy.cpp -o np_fft`python3-config --extension-suffix`
