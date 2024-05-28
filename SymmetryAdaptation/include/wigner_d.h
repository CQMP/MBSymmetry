#ifndef SYMMETRYADAPTATION_WIGNER_D_H
#define SYMMETRYADAPTATION_WIGNER_D_H

#include "type.h"
#include "utils.h"
#include "group_base.h"

namespace symmetry_adaptation {

static std::vector<Matrix<dcomplex> > Ylm_im2re_transform_matrix(const std::string &convention = "pyscf") {

  static constexpr std::complex<double> I_{std::complex<double>(0., 1.)};

  if (convention == "pyscf") {
    auto T = std::vector<Matrix<dcomplex> > {
      Eigen::Matrix<std::complex<double>, 1, 1> {1},
      // l = 1
      (Eigen::Matrix<std::complex<double>, 3, 3> () <<
        -1./std::sqrt(2), 0., 1./std::sqrt(2),
        -I_/std::sqrt(2), 0., -I_/std::sqrt(2),
        0., 1., 0.
      ).finished(),
      // l = 2
      (Eigen::Matrix<std::complex<double>, 5, 5> () <<
        I_/std::sqrt(2), 0., 0., 0., -I_/std::sqrt(2),
        0., -I_/std::sqrt(2), 0., -I_/std::sqrt(2), 0.,
        0., 0., 1., 0., 0.,
        0., -1./std::sqrt(2), 0., 1./std::sqrt(2), 0.,
        1./std::sqrt(2), 0., 0., 0., 1./std::sqrt(2)
      ).finished(),
      // l = 3
      (Eigen::Matrix<std::complex<double>, 7, 7> () <<
        -I_/std::sqrt(2), 0., 0., 0., 0., 0., -I_/std::sqrt(2),
        0., I_/std::sqrt(2), 0., 0., 0., -I_/std::sqrt(2), 0.,
        0., 0., -I_/std::sqrt(2), 0., -I_/std::sqrt(2), 0., 0.,
        0., 0., 0., 1., 0., 0., 0.,
        0., 0., -1./std::sqrt(2), 0., 1./std::sqrt(2), 0., 0.,
        0., 1./std::sqrt(2), 0., 0., 0., 1./std::sqrt(2), 0.,
        -1./std::sqrt(2), 0., 0., 0., 0., 0., 1./std::sqrt(2)
      ).finished(),
      // l = 4
      (Eigen::Matrix<std::complex<double>, 9, 9> () <<
          I_/std::sqrt(2), 0., 0., 0., 0., 0., 0., 0., -I_/std::sqrt(2),
          0., -I_/std::sqrt(2), 0., 0., 0., 0., 0., -I_/std::sqrt(2), 0.,
          0., 0., I_/std::sqrt(2), 0., 0., 0., -I_/std::sqrt(2), 0., 0.,
          0., 0., 0., -I_/std::sqrt(2), 0., -I_/std::sqrt(2), 0., 0., 0.,
          0., 0., 0., 0., 1., 0., 0., 0., 0.,
          0., 0., 0., -1./std::sqrt(2), 0., 1./std::sqrt(2), 0., 0., 0.,
          0., 0., 1./std::sqrt(2), 0., 0., 0., 1./std::sqrt(2), 0., 0.,
          0., -1./std::sqrt(2), 0., 0., 0., 0., 0., 1./std::sqrt(2), 0.,
          1./std::sqrt(2), 0., 0., 0., 0., 0., 0., 0., 1./std::sqrt(2)
      ).finished(),
    };
    return T;
  }
  else {
    throw std::runtime_error("only support pyscf convention");
  }
}

class WignerD {
public:
  WignerD(int n=0, const std::string &convention = "pyscf",
          double tol=1e-6): n_(n), h_(0), convention_(convention), tol_(tol),
          T_im2re_(Ylm_im2re_transform_matrix(convention)) {};
  ~WignerD() = default;

  static constexpr int max_l = 5;

  // Get point group number
  inline int n() const { return n_; }

  // Get order of the group
  inline int h() const { return h_; }
  inline int order() const { return h_; }

  inline const std::string &convention() const { return convention_; }

  inline const Matrix<dcomplex> &D(int l, int g) const { return D_[l][g]; }

  inline const std::vector<Matrix<dcomplex> > &D(int l) const { return D_[l]; }

  void compute(GroupBase &group);

  void save(alps::hdf5::archive &ar, const std::string &group = "") const;

private:

  int n_;
  int h_;
  std::string convention_;
  double tol_;

  // Dimension: l, g, mat
  std::vector<std::vector<Matrix<dcomplex> > > D_;

  // Transformation matrices from complex spherical to real spherical. Unitary matrices
  const std::vector<Matrix<dcomplex> > T_im2re_;

  /*
   * Methods needed for computing D matrices
   */
  static RowVector<double, DIM> compute_Euler_angles_zyz(const Matrix<double, DIM, DIM> &space_rep,
                                                         bool is_proper, double tol);

  static Matrix<double> compute_Wigner_small_d_matrix_y(int l, double beta);

  static Matrix<dcomplex> compute_Wigner_D_matrix_zyz(int l, const RowVector<double, DIM> &angles, bool is_proper);
};

} // namespace symmetry_adaptation


#endif //SYMMETRYADAPTATION_WIGNER_D_H
