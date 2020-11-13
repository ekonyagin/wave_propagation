#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
    
#include <algorithm>
#include <iostream>
#include <cmath> 


// -------------
// pure C++ code
// -------------

inline double window_pos(int idx, double dtau_window, int zmin){
    return zmin + idx * dtau_window;
}

void apply_filter(std::vector<double>& z, int idx){
    double zmin = -15., zmax = 15., dtau_window = 0.1, window_width = 2.0;
    std::cout << z[0] << std::endl;    
    //auto print = [](const double& n) { std::cout << " " << n; };
    double winPos = window_pos(idx, dtau_window, z[0]);

    std::cout << winPos<<std::endl;
    std::for_each(z.begin(), z.end(), [&winPos, &window_width](double &n){ n=exp(-pow((n-winPos)/window_width, 100.)); });
    
    //Filter(dtau_window, window_width, idx, z);

    //std::for_each(z.cbegin(), z.cend(), print);
    //std::cout << '\n';
    //return z_exp;
}

std::vector<double> apply_sliding_window(const std::vector<double>& input, int idx)
{
  std::vector<double> output(input.size());

  for ( size_t i = 0 ; i < input.size() ; ++i )
    output[i] = static_cast<double>(input[i]);

  apply_filter(output, idx);
  return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO

//py::array_t<double> py_multiply(py::array_t<double, py::array::c_style | py::array::forcecast> array, int idx)
py::array_t<double> create_data(int idx, int size, int zmin, int zmax)
{
  // allocate std::vector (to pass to the C++ function)
  std::vector<double> array_vec(size);

  // copy py::array -> std::vector
  //std::memcpy(array_vec.data(),array.data(),array.size()*sizeof(double));
  
  double step = (double)(zmax-zmin)/(double)size;
  
  for(size_t i=0; i<size; i++)
    array_vec[i] = (double)zmin + step*i;
  // call pure C++ function
  std::vector<double> result_vec = apply_sliding_window(array_vec, idx);

  // allocate py::array (to pass the result of the C++ function to Python)
  auto result        = py::array_t<double>(array_vec.size());
  auto result_buffer = result.request();
  double *result_ptr = (double *) result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr,result_vec.data(),result_vec.size()*sizeof(double));

  return result;
}

// wrap as Python module
PYBIND11_MODULE(example,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("window_func", &create_data, "Arguments: (idx, size, zmin, zmax)");
}