#include <fstream>
#include <string>

#include "transformer_t.h"

namespace symmetry_mbpt {
void transformer_t::read_trans_ir_statistics(alps::hdf5::archive &tnl_file, int eta,
                                             MatrixXcd &Tnc_out,
                                             MatrixXcd &Tcn_out,
                                             MatrixXd &Ttc_out,
                                             MatrixXd &Tct_out) {
  std::string prefix = eta ? "fermi" : "bose";
  size_t nts = eta ? nts_ : nts_b_;
  // Tnc_ = (nw, nl), Ttc_ = (nx+2, nl), Tct_ = (nl, nx), Ttc_other_ = (nx_b+2, nl)
  size_t ncheb, nw;

  tnl_file[prefix + "/nx"] >> ncheb;
  tnl_file[prefix + "/nw"] >> nw;
  assert(ncheb == (eta ? ncheb_ : ncheb_b_));

  Tnc_out = MatrixXcd::Zero(nw, ncheb);
  MatrixXd Tnc_r = MatrixXd::Zero(nw, ncheb);
  MatrixXd Tnc_i = MatrixXd::Zero(nw, ncheb);
  Ttc_out = MatrixXd::Zero(nts, ncheb);
  MatrixXd Ttc_tmp = MatrixXd::Zero(ncheb, ncheb);
  MatrixXd Tct_tmp = MatrixXd::Zero(ncheb, ncheb);
  std::vector<double> Tt1c_tmp(ncheb);
  std::vector<double> Tt1c_minus_tmp(ncheb);
  tnl_file.read(prefix + "/uwl_re", Tnc_r.data(), std::vector<size_t>({nw, ncheb}));
  tnl_file.read(prefix + "/uwl_im", Tnc_i.data(), std::vector<size_t>({nw, ncheb}));
  tnl_file.read(prefix + "/uxl", Ttc_tmp.data(), std::vector<size_t>({ncheb, ncheb}));
  tnl_file.read(prefix + "/ux1l", Tt1c_tmp.data(), std::vector<size_t>({ncheb}));
  tnl_file.read(prefix + "/ux1l_minus", Tt1c_minus_tmp.data(), std::vector<size_t>({ncheb}));
  Tnc_out.real() = Tnc_r;
  Tnc_out.imag() = Tnc_i;
  Tcn_out = Tnc_out.completeOrthogonalDecomposition().pseudoInverse();
  Tct_out = Ttc_tmp.completeOrthogonalDecomposition().pseudoInverse();

  for (size_t ic = 0; ic < ncheb; ++ic) {
    Ttc_out(0, ic) = Tt1c_minus_tmp[ic];
    Ttc_out(nts - 1, ic) = Tt1c_tmp[ic];
    for (size_t it = 0; it < ncheb; ++it) {
      Ttc_out(it + 1, ic) = Ttc_tmp(it, ic);
    }
  }
}

void transformer_t::read_trans_ir(const std::string &path) {
  alps::hdf5::archive tnl_file(path);
  read_trans_ir_statistics(tnl_file, 1, Tnc_, Tcn_, Ttc_, Tct_);
  read_trans_ir_statistics(tnl_file, 0, Tnc_B_, Tcn_B_, Ttc_B_, Tct_B_);

  MatrixXd Ttc_other_tmp = MatrixXd::Zero(ncheb_b_, ncheb_);
  MatrixXd Ttc_B_other_tmp = MatrixXd::Zero(ncheb_, ncheb_b_);
  tnl_file.read("/fermi/other_uxl", Ttc_other_tmp.data(), std::vector<size_t>({size_t(ncheb_b_), size_t(ncheb_)}));
  tnl_file.read("/bose/other_uxl", Ttc_B_other_tmp.data(), std::vector<size_t>({size_t(ncheb_), size_t(ncheb_b_)}));

  // (nts_b_, ncheb_b_) * (ncheb_b_, ncheb_b_) * (ncheb_b_, ncheb_) = (nts_b_, nl)
  Ttc_other_ = Ttc_B_ * Tct_B_ * Ttc_other_tmp;
  Ttc_B_other_ = Ttc_ * Tct_ * Ttc_B_other_tmp;
  tnl_file.close();


  Ttc_ *= std::sqrt(2.0 / beta_);
  Ttc_B_ *= std::sqrt(2.0 / beta_);
  Ttc_other_ *= std::sqrt(2.0 / beta_);
  Ttc_B_other_ *= std::sqrt(2.0 / beta_);
  Tct_ *= std::sqrt(beta_ / 2.0);
  Tct_B_ *= std::sqrt(beta_ / 2.0);
  Tnc_ *= std::sqrt(beta_);
  Tnc_B_ *= std::sqrt(beta_);
  Tcn_ *= std::sqrt(1.0 / beta_);
  Tcn_B_ *= std::sqrt(1.0 / beta_);

  Tnt_ = Tnc_ * Tct_;
  Ttn_ = Ttc_ * Tcn_;
  Tnt_B_ = Tnc_B_ * Tct_B_;
  Ttn_B_ = Ttc_B_ * Tcn_B_;

  Tnt_BF_ = Tnc_B_ * Tct_B_ * Ttc_other_.block(1, 0, ncheb_b_, ncheb_) * Tct_;
  Ttn_FB_ = Ttc_B_other_ * Tcn_B_;
}

void transformer_t::read_chebyshev(const std::string &path, const std::string &path_B) {
  // Read Tnc_
  alps::hdf5::archive tnl_file(path);
  size_t nwn = ncheb_;
  Tnc_ = MatrixXcd::Zero(nwn, ncheb_);
  for (size_t ic = 0; ic < ncheb_; ++ic) {
    std::vector<double> re;
    std::vector<double> im;
    tnl_file["fermi/awl_" + std::to_string(ic) + "_r"] >> re;
    tnl_file["fermi/awl_" + std::to_string(ic) + "_i"] >> im;
    for (size_t n = 0; n < nwn; ++n) {
      Tnc_(n, ic) = std::complex<double>(re[n], im[n]) * beta_ / 2.0;
    }
  }
  tnl_file.close();
  // Read Tnc_B_
  alps::hdf5::archive tnl_B_file(path_B);
  size_t nwn_B = ncheb_b_;
  Tnc_B_ = MatrixXcd::Zero(nwn_B, ncheb_b_);
  for (size_t ic = 0; ic < ncheb_b_; ++ic) {
    std::vector<double> re;
    std::vector<double> im;
    tnl_B_file["bose/awl_" + std::to_string(ic) + "_r"] >> re;
    tnl_B_file["bose/awl_" + std::to_string(ic) + "_i"] >> im;
    for (size_t n = 0; n < nwn_B; ++n) {
      Tnc_B_(n, ic) = std::complex<double>(re[n], im[n]) * beta_ / 2.0;
    }
  }
  tnl_B_file.close();

  Tcn_ = Tnc_.inverse();
  Tcn_B_ = Tnc_B_.inverse();
}

void transformer_t::read_wsample(const std::string &path, const std::string &path_b, std::vector<long> &wsample,
                                 std::vector<long> &wsample_b) {
  alps::hdf5::archive sample_file(path);
  sample_file["fermi/wsample"] >> wsample;
  sample_file.close();
  alps::hdf5::archive sample_file_b(path_b);
  sample_file_b["bose/wsample"] >> wsample_b;
  sample_file_b.close();
}

void transformer_t::init_chebyshev_grid() {
  // Define Chebyshev polinomials (i.e Ttc_) on Fermionic grid
  Ttc_ = MatrixXd::Zero(nts_, ncheb_);
  for (size_t it = 0; it < nts_; ++it) {
    // map tau-point onto [-1;1]
    double x = 2.0 * tau_mesh_[it] / beta_ - 1.0;
    // 0-th Chebyshev polynomial is constant
    Ttc_(it, 0) = 1.0;
    // 1-st Chebyshev polynomial is 'x'
    Ttc_(it, 1) = x;
    // Use recurrence relation to compute higher order Chebyshev polynomials
    for (size_t ic = 2; ic < ncheb_; ++ic) {
      Ttc_(it, ic) = 2.0 * x * Ttc_(it, ic - 1) - Ttc_(it, ic - 2);
    }
  }
  // Normalization factor
  double x = 1. / (ncheb_);
  // Evaluate transition matrix between imaginary time and Chebyshev polynomials spaces
  Tct_ = MatrixXd::Zero(ncheb_, nts_);
  for (size_t ic(0); ic < ncheb_; ++ic) {
    // we have different normalization for 0-th Chebyshev coefficient
    double factor = ic == 0 ? 1.0 : 2.0;
    for (size_t it(1); it < nts_ - 1; ++it) {
      // Here "it's" are defined at the Chebyshev nodes, which allow us to exploit discrete orthogonality properties.
      Tct_(ic, it) = Ttc_(it, ic) * factor * x;
    }
  }
  // Define Chebyshev polynomials (i.e. Ttc_) on Fermionic grid
  Ttc_B_ = MatrixXd::Zero(nts_b_, ncheb_b_);
  for (size_t it = 0; it < nts_b_; ++it) {
    double x = 2.0 * tau_mesh_B_[it] / beta_ - 1.0;
    // 0-th Chebyshev polynomial is constant
    Ttc_B_(it, 0) = 1.0;
    // 1-st Chebyshev polynomial is 'x'
    Ttc_B_(it, 1) = x;
    // Use recurrence relation to compute higher order Chebyshev polynomials
    for (size_t ic = 2; ic < ncheb_b_; ++ic) {
      Ttc_B_(it, ic) = 2.0 * x * Ttc_B_(it, ic - 1) - Ttc_B_(it, ic - 2);
    }
  }
  x = 1. / ncheb_b_;
  // Evaluate transition matrix between imaginary time and Chebyshev polynomials spaces
  Tct_B_ = MatrixXd::Zero(ncheb_b_, nts_b_);
  for (size_t ic = 0; ic < ncheb_b_; ++ic) {
    // we have different normalization for 0-th Chebyshev coefficient
    double factor = ic == 0 ? 1.0 : 2.0;
    for (size_t it = 1; it < nts_b_ - 1; ++it) {
      // Here "it's" are defined at the Chebyshev nodes, which allow us to exploit discrete orthogonality properties.
      Tct_B_(ic, it) = Ttc_B_(it, ic) * factor * x;
    }
  }
}

void transformer_t::fermi_boson_trans_ir(const tensor<dcomplex, 4> &F_t_before, tensor<dcomplex, 4> &F_t_after,
                                         int eta) {
  size_t dim_t = F_t_before.shape()[1] * F_t_before.shape()[2] * F_t_before.shape()[3];
  size_t dim_c = F_t_after.shape()[1] * F_t_after.shape()[2] * F_t_after.shape()[3];
  assert(dim_t == dim_c);

  if (eta == 1) {
    tensor<dcomplex, 4> F_c_before(ncheb_, F_t_before.shape()[1], F_t_before.shape()[2], F_t_before.shape()[3]);
    tau_to_chebyshev(F_t_before, F_c_before, eta);
    // Transform to other stat's grid
    MMatrixXcd f_t(F_t_after.data(), nts_b_, dim_t);
    //MMatrixXcd f_t(F_t_after.data()+dim1, ncheb_b_, dim1);
    CMMatrixXcd f_c(F_c_before.data(), F_c_before.shape()[0], dim_t);
    // Ttc_other_ = (nts_b_, ncheb_)
    f_t = Ttc_other_ * f_c;
  } else {
    tensor<dcomplex, 4> F_c_before(ncheb_b_, F_t_before.shape()[1], F_t_before.shape()[2], F_t_before.shape()[3]);
    tau_to_chebyshev(F_t_before, F_c_before, eta);
    // Transform to other stat's grid
    MMatrixXcd f_t(F_t_after.data(), nts_, dim_t);
    //MMatrixXcd f_t(F_t_after.data()+dim1, ncheb_, dim1);
    CMMatrixXcd f_c(F_c_before.data(), F_c_before.shape()[0], dim_t);
    // Ttc_B_other_ = (nts_, ncheb_b)
    f_t = Ttc_B_other_ * f_c;
  }
}

void transformer_t::fermi_boson_trans_cheby(const tensor<dcomplex, 4> &F_t_before, tensor<dcomplex, 4> &F_t_after,
                                            int eta) {
  // ncheb_ must be even number for Fermionic functions
  size_t ncheb_before = ncheb_ - (1 - eta);
  size_t ncheb_after = ncheb_ - eta;
  //F_t_after.reshape(ncheb_after+2, _nk, _nao, _nao);
  tensor<dcomplex, 4> F_c_before(ncheb_before, F_t_before.shape()[1], F_t_before.shape()[2], F_t_before.shape()[3]);
  // Tau to Chebyshev in the initial grid
  tau_to_chebyshev(F_t_before, F_c_before, eta);
  // Define Chebysheb polynomials on the final grid
  size_t nts_after = ncheb_after + 2;
  MatrixXd Tn = MatrixXd::Zero(nts_after, ncheb_before);
  // Construct Chebyshev polynomials on the new grids
  for (size_t it = 0; it < nts_after; ++it) {
    // map the new tau grid onto [-1;1]
    double x = (eta == 1) ? 2.0 * tau_mesh_B_[it] / beta_ - 1.0 : 2.0 * tau_mesh_[it] / beta_ - 1.0;
    // 0-th Chebyshev polynomial is constant
    Tn(it, 0) = 1.0;
    // 1-st Chebyshev polynomial is 'x'
    Tn(it, 1) = x;
    // Use recurrence relation to compute higher order Chebyshev polynomials
    for (size_t ic = 2; ic < ncheb_before; ++ic) {
      Tn(it, ic) = 2.0 * x * Tn(it, ic - 1) - Tn(it, ic - 2);
    }
  }
  // (ncheb)-th order to (ncheb-1)-th order
  size_t dim1 = F_t_before.shape()[1] * F_t_before.shape()[2] * F_t_before.shape()[3];
  CMMatrixXcd f_c(F_c_before.data(), F_c_before.shape()[0], dim1);
  MMatrixXcd f_t(F_t_after.data(), F_t_after.shape()[0], dim1);
  f_t = Tn * f_c;
}

} // namespace symmetry_mbpt