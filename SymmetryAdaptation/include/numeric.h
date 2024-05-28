#ifndef SYMMETRYADAPTATION_NUMERIC_H
#define SYMMETRYADAPTATION_NUMERIC_H

#include <vector>
#include <complex>

#include "type.h"


namespace lapack{
extern "C"  {
void zgeev_( char* jobvl, char* jobvr, int* n, std::complex<double> * a, int* lda, std::complex<double>* w,
             std::complex<double>* vl, int* ldvl, std::complex<double>* vr, int* ldvr,
             std::complex<double>* work, int* lwork, double* rwork, int* info );
}
}

namespace symmetry_adaptation {

namespace numeric {

// Return eigenvector matrix with descending eigenvalues
std::pair<ColVector<dcomplex>, Matrix<dcomplex> > eigenval_decomp(const Matrix<dcomplex> &A,
                                                                  double tol=1e-12, bool use_lapack=false);

// Gram Schmidt orthogonalization
template <typename T, int Row, int Col, int M>
void gram_schmidt(Matrix<T, Row, Col, M> &A, double tol = 1e-12) {
  int N = A.rows();

  ColVector<T> vec(N);
  ColVector<T> proj(N);
  for (int i = 1; i < N; ++i) {
    vec = A.col(i);
    for (int j = 0; j < i; ++j) {
      proj = (A.col(j).dot(vec)) * A.col(j);
      A.col(i) = A.col(i) - proj;
    }
    A.col(i) /= A.col(i).norm();
  }
  double error = (A.adjoint() * A - Matrix<T, Row, Col, M>::Identity(N, N)).norm() / A.norm();
  if (error > tol) {
    std::cout << "tolerance: " << tol << std::endl;
    std::cout << "error: " << error << std::endl;
    std::cout << (A.adjoint() * A).real() << std::endl;
    std::cout << std::endl;
    std::cout << (A.adjoint() * A).imag() << std::endl;
    throw std::runtime_error("Gram Schmidt fail.");
  }
}

// Modified Gram Schmidt orthogonalization
template <typename T, int Row, int Col, int M>
void modified_gram_schmidt(Matrix<T, Row, Col, M> &A, double tol = 1e-12) {
  int N = A.rows();

  ColVector<T> vec(N);
  ColVector<T> proj(N);
  for (int i = 0; i < N; ++i) {
    vec = A.col(i) / A.col(i).norm();
    for (int j = i+1; j < N; ++j) {
      proj = (vec.dot(A.col(j))) * vec;
      A.col(j) = A.col(j) - proj;
    }
    A.col(i) /= A.col(i).norm();
  }
  double error = (A.adjoint() * A - Matrix<T, Row, Col, M>::Identity(N, N)).norm() / A.norm();
  if (error > tol) {
    std::cout << "tolerance: " << tol << std::endl;
    std::cout << "error: " << error << std::endl;
    std::cout << (A.adjoint() * A).real() << std::endl;
    std::cout << std::endl;
    std::cout << (A.adjoint() * A).imag() << std::endl;
    throw std::runtime_error("Modified Gram Schmidt fail.");
  }
}

template <typename T, int Row, int Col, int M>
void householder_qr(Matrix<T, Row, Col, M> &A, double tol = 1e-12) {
  int N = A.rows();

  Eigen::HouseholderQR<Matrix<T, Row, Col, M> > qr(A);
  A = qr.householderQ();
  double error = (A.adjoint() * A - Matrix<T, Row, Col, M>::Identity(N, N)).norm() / A.norm();
  if (error > tol) {
    std::cout << "tolerance: " << tol << std::endl;
    std::cout << "error: " << error << std::endl;
    std::cout << (A.adjoint() * A).real() << std::endl;
    std::cout << std::endl;
    std::cout << (A.adjoint() * A).imag() << std::endl;
    throw std::runtime_error("QR orthogonolization fail.");
  }
}

inline double factorial(int n) {
  if (n > 1)
    return n * factorial(n - 1);
  if (n < 0)
    throw std::logic_error("can not compute factorial of a negative number");
  return 1;
}

template <typename T, int Row, int Col, int M>
bool is_in_span(const ColVector<T> &Ui, const Eigen::Block<Matrix<T, Row, Col, M> > &U, double tol = 1e-9) {
  // Check whether Ui is in the span of columns of U
  auto coeff = ((Ui.adjoint() * U).array() / U.colwise().squaredNorm().array()).matrix();
  auto ref_val = U * coeff.adjoint();
  if ((ref_val - Ui).norm() < tol)
    return true;
  std::cout << (ref_val - Ui).norm() << std::endl;
  return false;
}

template <typename T>
void get_orthonormal_basis(std::vector<ColVector<T> > &U_list, double tol=1e-9) {

  int N = U_list.size();
  int dim = U_list[0].size();

  ColVector<T> vec(dim);
  ColVector<T> proj(dim);
  U_list[0] = U_list[0] / U_list[0].norm();
  for (int i = 1; i < N; ++i) {
    vec = U_list[i];
    for (int j = 0; j < i; ++j) {
      proj = (U_list[j].dot(vec)) * U_list[j];
      U_list[i] = U_list[i] - proj;
    }
    if (U_list[i].norm() > tol)
      U_list[i] /= U_list[i].norm();
    else
      U_list[i].setZero();
  }
  std::vector<ColVector<T> > U_remain;
  for (int i = 0; i < N; ++i) {
    if (U_list[i].cwiseAbs().maxCoeff() > tol)
      U_remain.push_back(U_list[i]);
  }
  U_list = U_remain;
}

template <typename T>
void get_orthonormal_basis_mgs(std::vector<ColVector<T> > &U_list, double tol=1e-9) {

  int N = U_list.size();
  int dim = U_list[0].size();

  ColVector<T> vec(dim);
  ColVector<T> proj(dim);

  for (int i = 0; i < N; ++i) {
    U_list[i] = U_list[i] / U_list[i].norm();
  }
  for (int i = 0; i < N; ++i) {
    vec = U_list[i] / U_list[i].norm();
    for (int j = i+1; j < N; ++j) {
      proj = (vec.dot(U_list[j])) * vec;
      U_list[j] = U_list[j] - proj;
    }
  }
  std::vector<ColVector<T> > U_remain;
  for (int i = 0; i < N; ++i) {
    if (U_list[i].norm() > tol)
      U_remain.push_back(U_list[i]);
  }
  U_list = U_remain;
}

template <typename T>
void get_orthonormal_basis_householder(std::vector<ColVector<T> > &U_list, double tol=1e-9) {

  int N = U_list.size();
  int dim = U_list[0].size();
  Matrix<T> U(dim, N);
  for(int i = 0; i < N; ++i)
    U.col(i) = U_list[i];
  Eigen::ColPivHouseholderQR<Matrix<T> > qr(U);  //HouseholderQR
  U = qr.householderQ();
  Matrix<T> R = qr.matrixQR().template triangularView<Eigen::Upper>();

  std::vector<ColVector<T> > U_remain;
  for(int i = 0; i < R.rows(); ++i) {
    if (R.row(i).cwiseAbs().maxCoeff() > tol)
      U_remain.push_back(U.col(i));
    else
      break;
  }

  U_list = U_remain;
}

} // namespace numeric

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_NUMERIC_H
