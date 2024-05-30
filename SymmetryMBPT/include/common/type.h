#ifndef SYMMETRYMBPT_TYPE_H
#define SYMMETRYMBPT_TYPE_H

#include <complex>
#include <Eigen/Dense>

#include <alps/numeric/tensors.hpp>

namespace symmetry_mbpt {

using dcomplex = std::complex<double>;

// Matrix aliases (we always use row major)
template <typename T, int Row=Eigen::Dynamic, int Col=Eigen::Dynamic, int M=Eigen::RowMajor>
using Matrix = Eigen::Matrix<T, Row, Col, M>;
template <typename T, int Row=Eigen::Dynamic, int Col=Eigen::Dynamic, int M=Eigen::ColMajor>
using CMatrix = Eigen::Matrix<T, Row, Col, M>;
template <typename T, int Col=Eigen::Dynamic>
using RowVector = Eigen::Matrix<T, 1, Col, Eigen::RowMajor>;
template <typename T, int Row=Eigen::Dynamic>
using ColVector = Eigen::Matrix<T, Row, 1, Eigen::ColMajor>;

template <typename T, int Row=Eigen::Dynamic, int Col=Eigen::Dynamic, int M=Eigen::RowMajor>
using MatrixMap = Eigen::Map<Matrix<T, Row, Col, M> >;
template <typename T, int Row=Eigen::Dynamic, int Col=Eigen::Dynamic, int M=Eigen::RowMajor>
using MatrixConstMap = Eigen::Map<const Matrix<T, Row, Col, M> >;

template <typename T, int Row=Eigen::Dynamic, int Col=Eigen::Dynamic, int M=Eigen::ColMajor>
using CMatrixMap = Eigen::Map<Matrix<T, Row, Col, M> >;
template <typename T, int Row=Eigen::Dynamic, int Col=Eigen::Dynamic, int M=Eigen::ColMajor>
using CMatrixConstMap = Eigen::Map<const Matrix<T, Row, Col, M> >;

template <typename T, int Col=Eigen::Dynamic>
using RowVectorMap = Eigen::Map<RowVector<T, Col> >;
template <typename T, int Col=Eigen::Dynamic>
using RowVectorConstMap = Eigen::Map<const RowVector<T, Col> >;

template <typename T, int Row=Eigen::Dynamic>
using ColVectorMap = Eigen::Map<ColVector<T, Row> >;
template <typename T, int Row=Eigen::Dynamic>
using ColVectorConstMap = Eigen::Map<const ColVector<T, Row> >;

template <typename T, int Col=Eigen::Dynamic>
using VectorMap = RowVectorMap<T, Col>;
template <typename T, int Col=Eigen::Dynamic>
using VectorConstMap = RowVectorConstMap<T, Col>;


// -- below are types defined to accommodate old implementations
template<typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = MatrixX<double>;
using MatrixXcd = MatrixX<dcomplex>;
template<typename T>
using MMatrixX = Eigen::Map<MatrixX<T> >;
using MMatrixXcd = MMatrixX<dcomplex>;
template<typename T>
using CMMatrixX = Eigen::Map<const MatrixX<T> >;
using CMMatrixXcd = CMMatrixX<dcomplex>;

template<typename T>
using column = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
using columnXcd = column<dcomplex>;
template<typename T>
using Mcolumn = Eigen::Map<column<T> >;
template<typename T>
using CMcolumn = Eigen::Map<const column<T> >;

template<typename T, size_t Dim>
using tensor = alps::numerics::tensor<T, Dim>;
template<typename T, size_t Dim>
using tensor_view = alps::numerics::tensor_view<T, Dim>;
template<typename T, size_t Dim>
using Ctensor_view = alps::numerics::tensor_view<const T, Dim>;
template<typename T, size_t Dim, typename S>
using tensor_base= alps::numerics::detail::tensor_base<T, Dim, S>;

enum solver_type_e {
  HF, GW, cuGW, cuHF,
};

enum self_consistency_type_e {
  Dyson
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_TYPE_H
