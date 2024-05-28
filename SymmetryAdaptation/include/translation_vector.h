#ifndef SYMMETRYADAPTATION_TRANSLATION_VECTOR_H
#define SYMMETRYADAPTATION_TRANSLATION_VECTOR_H

#include "type.h"

namespace symmetry_adaptation {

class TranslationVector {

public:

  explicit TranslationVector(const std::string &tvec_string, bool verbose=true) {
    read_a_vectors(tvec_string, verbose);
    compute_b_vectors(verbose);
  };

  const Matrix<double, DIM, DIM> &a_vectors() const { return T_; }
  auto a(int n) const { return T_.col(n); }

  const Matrix<double, DIM, DIM> &a_vectors_inv() const { return T_inv_; }

  const Matrix<double, DIM, DIM> &b_vectors() const { return B_; }
  auto b(int n) const { return B_.col(n); }

  const Matrix<double, DIM, DIM> &b_vectors_inv() const { return B_inv_; }

  ColVector<double, DIM> shift_back_to_center_cell(const ColVector<double, DIM> &vec, double tol=1e-6) const;

private:

  void read_a_vectors(const std::string &xyz_string, bool verbose=true);
  void compute_b_vectors(bool verbose=true);

  Matrix<double, DIM, DIM> T_; // row major matrix with each column being one lattice vector
  Matrix<double, DIM, DIM> B_; // row major matrix with each column being one reciprocal lattice vector

  Matrix<double, DIM, DIM> T_inv_;
  Matrix<double, DIM, DIM> B_inv_;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_TRANSLATION_VECTOR_H
