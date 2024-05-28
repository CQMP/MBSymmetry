
#include <iostream>

#include "simul_diag.h"
#include "numeric.h"

namespace symmetry_adaptation {

Matrix<dcomplex> SimulDiag::solve(const std::vector<Matrix<dcomplex> > &ops, double tol, bool use_lapack) {

  validate_matrices(ops, tol);

  const Matrix<dcomplex> &A = ops[0];

  auto EU = numeric::eigenval_decomp(A, tol, use_lapack);
  auto E = EU.first;
  auto U = EU.second;
  int N = E.size();

  int k = 0;
  while (k < N) {
    int i = k + 1;
    while (i < N) {
      if (std::abs(E[i].real() - E[k].real()) > std::max(tol, tol * std::abs(E[k].real()))
          || std::abs(E[i].imag() - E[k].imag()) > std::max(tol, tol * std::abs(E[k].imag()))) {
        break;
      }
      ++i;
    }
    // have degeneracy
    if (i != k + 1){
      U.block(0, k, N, i - k) = degen(ops, U.block(0, k, N, i - k), 1, tol);
    }
    k = i;
  }

  // normalize
  for (int i = 0; i < N; ++i) {
    U.col(i) = U.col(i) / U.col(i).norm();
  }

  double unitary_diff = (U.adjoint() * U - Matrix<dcomplex>::Identity(N, N)).norm() / U.norm();
  if (unitary_diff > tol) {
    std::cout << "Performing final orthogonalization" << std::endl;
    numeric::householder_qr(U, tol);
    //numeric::gram_schmidt(U, tol);
    //numeric::modified_gram_schmidt(U, tol);
  }

  check_diag(ops, U, tol);

  return U;
}

void SimulDiag::validate_matrices(const std::vector<Matrix<dcomplex> > &ops, double tol) {

  int N = ops.size();
  int shape = ops[0].cols();
  for (int i = 0; i < N; ++i) {
    auto &A = ops[i];
    if (A.rows() != A.cols()) throw std::runtime_error("Matrices must be square.");
    if (A.rows() != shape) throw std::runtime_error("All matrices must have same shape.");

    double norm_diff = (A.adjoint() * A - A * A.adjoint()).norm() / A.norm();
    if (norm_diff > tol) throw std::runtime_error("All matrices must be Normal.");

    for (int j = 0; j < N; ++j) {
      auto &B = ops[j];
      double commutator = (A * B - B * A).norm();
      if (commutator > tol && A.norm() > tol && B.norm() > tol) {
        std::cout << A.norm() << " " << B.norm() << " " << commutator << std::endl;
        throw std::runtime_error("All matrices must commute.");
      }
    }
  }
}

Matrix<dcomplex> SimulDiag::degen(const std::vector<Matrix<dcomplex>> &ops, const Eigen::Block<Matrix<dcomplex>> &U_in,
                                  int start_idx, double tol, bool use_lapack) {

  int N = ops.size();
  if (start_idx == N) {
    auto U = U_in;
    return U;
  }

  const Matrix<dcomplex> &A = ops[start_idx];
  auto X = U_in.adjoint() * (A * U_in);

  auto EU = numeric::eigenval_decomp(X, tol, use_lapack);
  auto E = EU.first;
  auto U = EU.second;

  N = E.size();

  Matrix<dcomplex> U_new = U_in * U;

  int k = 0;
  while (k < N) {
    int i = k + 1;
    while (i < N) {
      if (std::abs(E[i].real() - E[k].real()) > std::max(tol, tol * std::abs(E[k].real()))
          || std::abs(E[i].imag() - E[k].imag()) > std::max(tol, tol * std::abs(E[k].imag()))) {
        break;
      }
      ++i;
    }
    // have degeneracy
    if (i != k + 1) {
      U_new.block(0, k, U_new.rows(), i - k) = degen(ops, U_new.block(0, k, U_new.rows(), i - k), start_idx+1, tol);
    }
    k = i;
  }

  return U_new;
}

void SimulDiag::check_diag(const std::vector<Matrix<dcomplex> > &ops,
                           const Matrix<dcomplex> &U, double tol) {

  Matrix<dcomplex> X(U.rows(), U.cols());
  Matrix<dcomplex> diag(U.rows(), U.cols());

  for (const auto & A : ops) {
    X = U.adjoint() * A * U;
    diag = X.diagonal().asDiagonal();
    double error = (X - diag).norm() / A.norm();
    double max_diff = (X - diag).cwiseAbs().maxCoeff();

    if (error > tol) {
      std::cout << "Error of simultaneous diagonalization is " << error << std::endl;
      throw std::runtime_error("Simultaneous diagonalization fail.");
    }
    if (max_diff > tol) std::cout <<
                                  "Warning: Absolute error of simultaneous diagonalization is "
                                  << max_diff << std::endl;
  }
}

} // namespace symmetry_adaptation
