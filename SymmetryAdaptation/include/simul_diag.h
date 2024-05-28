#ifndef SYMMETRYADAPTATION_SIM_DIAG_H
#define SYMMETRYADAPTATION_SIM_DIAG_H

#include "type.h"

namespace symmetry_adaptation {

class SimulDiag{
public:
  SimulDiag() = default;
  ~SimulDiag() = default;

  static Matrix<dcomplex> solve(const std::vector<Matrix<dcomplex> > &ops, double tol=1e-12, bool use_lapack=true);

  // Check all matrices are Hermitian and commute with each other
  static void validate_matrices(const std::vector<Matrix<dcomplex> > &ops, double tol=1e-12);

  static void check_diag(const std::vector<Matrix<dcomplex> > &ops,  const Matrix<dcomplex> &U, double tol=1e-12);

private:

  // Finds eigen values and vectors for degenerate matrices
  static Matrix<dcomplex> degen(const std::vector<Matrix<dcomplex> > &ops, const Eigen::Block<Matrix<dcomplex> > &U_in,
                                int start_idx, double tol=1e-12, bool use_lapack=true);
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_SIM_DIAG_H
