
#include <iostream>
#include "numeric.h"

namespace symmetry_adaptation {

namespace numeric {

std::pair<ColVector<dcomplex>, Matrix<dcomplex> > eigenval_decomp(const Matrix<dcomplex> &A, double tol,
                                                                  bool use_lapack) {

  int size = A.rows();

  ColVector<dcomplex> E(size);
  Matrix<dcomplex> temp(size, size);
  CMatrix<dcomplex> U(size, size);

  // Unstable in some cases
  if (!use_lapack) {
    Eigen::ComplexEigenSolver<Matrix<dcomplex> > es(A);

    E = es.eigenvalues();
    U = es.eigenvectors();
  }

  else {
    temp = A.transpose(); // row major -> column major

    int wsize = 4 * size * size;
    std::vector<std::complex<double> > lwork(wsize);
    std::vector<double> rwork(wsize);
    char Vec = 'V';
    char NoVec = 'N';
    int info = 0;
    CMatrix<dcomplex> ignored(size, size);

    // Make sure to pass in matrices for left and right eigenvectors
    // Lapack zgeev deals with column major matrix
    lapack::zgeev_(&NoVec, &Vec, &size, &(temp(0, 0)), &size, &(E[0]),
                   &(ignored(0, 0)), &size, &(U(0, 0)), &size, &(lwork[0]), &wsize, &(rwork[0]), &info);

    if (info != 0) throw std::runtime_error("lapack zgeev fail");
  }
  // sort eigenvalues
  std::vector<int> V(size);
  std::iota(V.begin(), V.end(), 0);
  auto compare = [&](int i, int j) {
    if (std::abs(E[i].real() - E[j].real()) < tol)
      return (E[i].imag() > E[j].imag());
    else
      return std::make_tuple(E[i].real(), E[i].imag()) > std::make_tuple(E[j].real(), E[j].imag());
  };
  std::sort(V.begin(), V.end(), compare);
  ColVector<dcomplex> E_out(size);
  for (int i = 0; i < size; ++i) {
    temp.col(i) = U.col(V[i]); // store eigenvectors in columns
    E_out(i) = E(V[i]);
  }

  // Not sure if needed
  double unitary_diff = (temp.adjoint() * temp - Matrix<dcomplex>::Identity(size, size)).norm() / temp.norm();
  if (unitary_diff > tol) {
    householder_qr(temp, tol);
    //gram_schmidt(temp, tol);
    //modified_gram_schmidt(temp, tol);
  }

  return std::pair<ColVector<dcomplex>, Matrix<dcomplex> > (E_out, temp);
}

} // namespace numeric

} // namespace symmetry_adaptation

