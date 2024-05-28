#ifndef SYMMETRYADAPTATION_TYPE_H
#define SYMMETRYADAPTATION_TYPE_H

#include <complex>
#include <Eigen/Dense>

#include <ndarray/ndarray.h>

namespace symmetry_adaptation {

using dcomplex = std::complex<double>;

static constexpr int DIM = 3;

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

template <typename T>
using Ndarray = ndarray::ndarray<T>;

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_TYPE_H
