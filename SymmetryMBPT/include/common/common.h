#ifndef SYMMETRYMBPT_COMMON_H
#define SYMMETRYMBPT_COMMON_H

namespace symmetry_mbpt {

inline double matmul_cost_dcomplex(int M,int N,int K) {
  //this depends on how you count, see
  //https://forums.developer.nvidia.com/t/how-to-compute-gflops-for-gemm-blas/20218/6
  return  (8.*M*N*K + 12.*M*N);
}

inline double matmul_cost_double(int M,int N,int K) {
  return N*M*(2.*K-1);
}

enum matmul_type {
  dreal, dcplx
};

inline double matmul_cost(int M,int N,int K, matmul_type type=dcplx) {
  switch (type) {
    case dcplx:
      return matmul_cost_dcomplex(M, N, K);
    case dreal:
      return matmul_cost_double(M, N, K);
    default:
      throw std::runtime_error("matmul type not recognized");
  }
}

enum integral_reading_type {
  chunks, as_a_whole
};

enum integral_symmetry_type_e {
  direct
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_COMMON_H
