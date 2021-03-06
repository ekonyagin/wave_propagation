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

inline double window_pos(int idx, double dtau_window, double zmin){
    return zmin + idx * dtau_window;
}

void create_filt(std::vector<double>& input, int idx, double zrange, int size)
{
  double dtau_window = 0.1, window_width = 2.0;
  double winPos = window_pos(idx, dtau_window, 0.0);
  double step = (double)(zrange)/(double)size;
  
  for(int i=0; i<size; i++)
      input[i] = 0.0 + step*i;

  std::for_each(input.begin(), input.end(), [&winPos, &window_width](double &n){ n=exp(-pow((n-winPos)/window_width, 100.)); });
}

namespace py = pybind11;

// wrap C++ function with NumPy array IO

//py::array_t<double> py_multiply(py::array_t<double, py::array::c_style | py::array::forcecast> array, int idx)
py::array_t<double> create_data(py::array_t<double, py::array::c_style | py::array::forcecast> array, 
                                int n_replicas, int size, double zrange, int stride)
{
  // allocate std::vector (to pass to the C++ function)
  
  std::vector<double> filt(size);
  //create_filt(filt, idx, zmin, zmax, size);
  
  //int n_replicas = 300;
  std::vector<double> result_vec(size * n_replicas * stride);

  for(int k=0; k<n_replicas; k++){
      create_filt(filt, k, zrange, size);
      for(int i=0; i<stride; i++){
          for(int j=0; j<size; j++){
              result_vec[k*stride*size + i*size + j] = /**(((double*)array.data())+j)**/ filt[j];
              if (k*stride*size + i*size + j<20)
                std::cout << *(((double*)array.data())+j) << "\n";
          }
      }
  }
  /*
  for(int i=0; i<n_replicas;i++){
      create_filt(filt, i, zrange, size);
      for(int j=0; j<size; j++){
          result_vec[i*size + j] = *(((double*)array.data())+j)*filt[j];
      }
  }
  */
  // allocate py::array (to pass the result of the C++ function to Python)
  auto result        = py::array_t<double>(result_vec.size());
  auto result_buffer = result.request();
  double *result_ptr = (double *) result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr,result_vec.data(),result_vec.size()*sizeof(double));

  return result;
}

// wrap as Python module
PYBIND11_MODULE(np_fft,m)
{
  m.doc() = "Tool for accelerated spectrogram counting";

  m.def("window_func", &create_data, "Arguments: (array - input 1D np_array,\n n_replicas - length of slice along Z-axis,\n size - length of a single Z-slice,\n zmin,\n zmax,\n stride - length of one r*phi stripe\n)");
}