#ifndef SYMMETRYMBPT_CUDA_TYPES_MAP_H
#define SYMMETRYMBPT_CUDA_TYPES_MAP_H

#ifdef WITH_CUDA

#include <complex>
#include <cuComplex.h>

template<typename T>
struct cu_type_map {};

template<typename cuT>
struct get_type_map {};

template<>
struct get_type_map<cuComplex> {
  using type_map = cu_type_map<std::complex<float>>;
};

template<>
struct get_type_map<cuDoubleComplex> {
  using type_map = cu_type_map<std::complex<double>>;
};

template<>
struct cu_type_map<std::complex<double>> {

  using cxx_base_type = double;
  using cxx_type = std::complex<double>;
  using cuda_type = cuDoubleComplex;

  static __inline__ cuda_type cast(cxx_type in) {
    return make_cuDoubleComplex(in.real(), in.imag());
  }

  static __inline__ cuda_type cast(cxx_base_type in_r, cxx_base_type in_i) {
    return make_cuDoubleComplex(in_r, in_i);
  }

  __host__ __device__ static __inline__ cxx_base_type real(cuda_type x) {
    return cuCreal(x);
  }

  __host__ __device__ static __inline__ cxx_base_type imag(cuda_type x) {
    return cuCimag(x);
  }

  __host__ __device__ static __inline__ cuda_type add(cuda_type x, cuda_type y) {
    return cuCadd(x, y);
  }

  __host__ __device__ static __inline__ cuda_type mul(cuda_type x, cuda_type y) {
    return cuCmul(x, y);
  }
};

template<>
struct cu_type_map<std::complex<float>> {

  using cxx_base_type = float;
  using cxx_type = std::complex<float>;
  using cuda_type = cuComplex;

  static __inline__ cuda_type cast(cxx_type in) {
    return make_cuComplex(in.real(), in.imag());
  }

  static __inline__ cuda_type cast(cxx_base_type in_r, cxx_base_type in_i) {
    return make_cuComplex(in_r, in_i);
  }

  __host__ __device__ static __inline__ cxx_base_type real(cuda_type x) {
    return cuCrealf(x);
  }

  __host__ __device__ static __inline__ cxx_base_type imag(cuda_type x) {
    return cuCimagf(x);
  }

  __host__ __device__ static __inline__ cuda_type add(cuda_type x, cuda_type y) {
    return cuCaddf(x, y);
  }

  __host__ __device__ static __inline__ cuda_type mul(cuda_type x, cuda_type y) {
    return cuCmulf(x, y);
  }
};

#endif

#endif //SYMMETRYMBPT_CUDA_TYPES_MAP_H
