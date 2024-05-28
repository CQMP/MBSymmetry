
#include "wigner_d.h"
#include "numeric.h"

namespace symmetry_adaptation {

void WignerD::save(alps::hdf5::archive &ar, const std::string &group) const {
  std::string prefix = group + "/WignerD/";

  for (int l = 0; l < max_l; ++l) {
    int dim = 2*l+1;
    Ndarray<dcomplex>D(h_, dim, dim);
    for(int g = 0; g < h_; ++g) {
      Ndarray_MatView(D, dim, dim, g*dim*dim) = D_[l][g];
    }
    ar[prefix + std::to_string(l)] << D;
  }
}

void WignerD::compute(GroupBase &group) {

  h_ = group.h();

  D_.resize(max_l);
  for (int l = 0; l < max_l; ++l) {
    D_[l].resize(h_);
  }
  RowVector<double, DIM> angles;
  for (int g = 0; g < h_; ++g) {
    bool is_proper = group.space_rep(g).determinant() > 0;
    angles = compute_Euler_angles_zyz(group.space_rep(g), is_proper, tol_);
    //std::cout << angles << std::endl;
    for (int l = 0; l < max_l; ++l) {
      // Transform to real spherical harmonics basis
      D_[l][g] = T_im2re_[l] * compute_Wigner_D_matrix_zyz(l, angles, is_proper) * T_im2re_[l].adjoint();
      if (D_[l][g].imag().norm() > tol_) {
        std::cout << "l: " << l  << " norm of im part: " << D_[l][g].imag().norm() << std::endl;
        throw std::runtime_error("Wigner D matrix for real spherical harmonics is not real");
      }
    }
  }
}

RowVector<double, DIM> WignerD::compute_Euler_angles_zyz(const Matrix<double, DIM, DIM> &space_rep,
                                                         bool is_proper, double tol) {

  Matrix<double, DIM, DIM> rep = is_proper ? space_rep : space_rep * (-1.);

  double alpha, beta, gamma;
  if (std::abs(rep(2, 0)) > tol || std::abs(rep(2, 1)) > tol) {
    alpha = std::atan2(rep(2, 1), rep(2, 0));
    gamma = std::atan2(rep(1, 2), -rep(0, 2));
    // Not sure if it's better to shift alpha and gamma to [0, 2*M_PI)
    alpha += alpha < 0 ? 2*M_PI : 0.;
    gamma += gamma < 0 ? 2*M_PI : 0.;

    beta = (std::abs(rep(2, 0)) > std::abs(rep(2, 1))) ? std::atan2(rep(2, 0) / std::cos(alpha), rep(2, 2))
                                                       : std::atan2(rep(2, 1) / std::sin(alpha), rep(2, 2));
  }
  else {
    alpha = std::atan2(rep(0, 1), rep(0, 0));
    beta = rep(2, 2) > 0 ? 0. : M_PI;
    gamma = beta;
  }

  return RowVector<double, DIM>{alpha, beta, gamma};
}

Matrix<dcomplex> WignerD::compute_Wigner_D_matrix_zyz(int l, const RowVector<double, DIM> &angles, bool is_proper) {

  dcomplex I = dcomplex{0, 1};
  Matrix<dcomplex> small_d_mat = compute_Wigner_small_d_matrix_y(l, angles[1]).cast<dcomplex>();
  RowVector<dcomplex> alpha_vec = (-I * angles[0] * ColVector<dcomplex>::LinSpaced(2*l+1, -l, l)).array().exp();
  ColVector<dcomplex> gamma_vec = (-I * angles[2] * ColVector<dcomplex>::LinSpaced(2*l+1, -l, l)).array().exp();

  Matrix<dcomplex> D_mat = (small_d_mat.array().rowwise() * alpha_vec.array()).colwise() * gamma_vec.array();
  if (!is_proper) {
    D_mat *= std::pow(-1, l);
  }

  return D_mat;
}

Matrix<double> WignerD::compute_Wigner_small_d_matrix_y(int l, double beta) {

  using namespace numeric;

  Matrix<double> d_mat = Matrix<double>::Zero(2*l+1, 2*l+1);

  double cos_beta = std::cos(beta/2);
  double sin_beta = std::sin(beta/2);
  for (int m1 = -l; m1 <= l; ++m1) {
    for (int m2 = -l; m2 <= l; ++m2) {
      int s_max = std::min(l - m1, l + m2);
      int s_min = std::max(0, m2 - m1);
      for (int s = s_min; s <= s_max; ++s) {
        d_mat(m1+l, m2+l) += std::pow(-1, s) * std::pow(cos_beta, 2*l+m2-m1-2*s) * std::pow(sin_beta, m1-m2+2*s)
                             / (factorial(s) * factorial(l-m1-s) * factorial(l+m2-s) * factorial(m1-m2+s));
      }
      d_mat(m1+l, m2+l) *= std::pow(-1, m1-m2)
        * std::sqrt(factorial(l+m2) * factorial(l-m2) * factorial(l+m1) * factorial(l-m1));
    }
  }

  return d_mat;
}

} // namespace symmetry_adaptation
