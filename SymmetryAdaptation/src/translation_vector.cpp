
#include <iostream>

#include "translation_vector.h"

namespace symmetry_adaptation {

void TranslationVector::read_a_vectors(const std::string &tvec_string, bool verbose) {

  std::stringstream sstr(tvec_string);

  for (int i = 0; i < DIM; ++i) {
    double tx, ty, tz;
    sstr >> tx >> ty >> tz;
    if (verbose)
      std::cout << "translation vector " << i << " : " << tx << " " << ty << " " << tz << std::endl;
    T_.col(i) << tx, ty, tz;
  }
  T_inv_ = T_.inverse();
}

void TranslationVector::compute_b_vectors(bool verbose) {

  double v = T_.col(0).dot(T_.col(1).cross(T_.col(2)));

  B_.col(0) = 2. * M_PI / v * T_.col(1).cross(T_.col(2));
  B_.col(1) = 2. * M_PI / v * T_.col(2).cross(T_.col(0));
  B_.col(2) = 2. * M_PI / v * T_.col(0).cross(T_.col(1));

  if (verbose) {
    for (int i = 0; i < DIM; ++i) {
      std::cout << "reciprocal lattice vector " << i << " : " << B_.col(i).transpose() << std::endl;
    }
  }
  B_inv_ = B_.inverse();
}

ColVector<double, DIM> TranslationVector::shift_back_to_center_cell(const ColVector<double, DIM> &vec,
                                                                    double tol) const {
  ColVector<double, DIM> scaled_vec = T_inv_ * vec;
  for (int i = 0; i < DIM; ++i) {
    while (scaled_vec[i] < -tol) {
      scaled_vec[i] += 1.;
    }
    while (scaled_vec[i] > 1 - tol) {
      scaled_vec[i] -= 1.;
    }
  }
  return T_ * scaled_vec;
}

} // namespace symmetry_adaptation
