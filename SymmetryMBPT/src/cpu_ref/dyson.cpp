
#include <alps/hdf5.hpp>
#include "dyson.h"

namespace symmetry_mbpt {

void dyson::read_input(const std::string &filename) {
  alps::hdf5::archive in_file(filename, "r");
  in_file["HF/Fock-k"] >> F_k_;
  in_file["HF/S-k"] >> S_k_;
  in_file["HF/H-k"] >> H_k_;
  in_file.close();
}

void dyson::solve(tensor_view<dcomplex, 5> &G_tskij, const tensor_view<dcomplex, 5> &Sigma_tskij, double mu,
                  const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
                  const column<dcomplex> &wsample) {
  compute_G(G_tskij, Sigma_tskij, F_k_, H_k_, S_k_, mu, T_wt, T_tw, wsample);
}

void dyson::compute_G(tensor_view<dcomplex, 5> &G_tskij, const tensor_view<dcomplex, 5> &Sigma_tskij,
                      const tensor<dcomplex, 4> &F_k, const tensor<dcomplex, 4> &H_k, const tensor<dcomplex, 4> &S_k,
                      double mu, const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
                      const column<dcomplex> &wsample) const {

  tensor<dcomplex, 3> G_t(nts_, nao_, nao_);
  tensor<dcomplex, 3> G_w(nw_, nao_, nao_);
  tensor<dcomplex, 3> Sigma_w(nw_, nao_, nao_);
  tensor<dcomplex, 3> Sigma_k(nts_, nao_, nao_);
  Eigen::FullPivLU<MatrixXcd> lusolver(nao_, nao_);

  G_tskij.set_zero();
  for (int s = 0; s < ns_; ++s) {
    for (int k = 0; k < nk_; ++k) {
      Sigma_k.set_zero();
      for (int t = 0; t < nts_; ++t) {
        Sigma_k(t).matrix() = CMMatrixX<dcomplex>(Sigma_tskij(t, s, k).data(), nao_, nao_);
      }
      MMatrixX<dcomplex>(Sigma_w.data(), nw_, nao_ * nao_)
        = T_wt * MMatrixX<dcomplex>(Sigma_k.data()+nao_*nao_, nw_, nao_ * nao_);

      CMMatrixX<dcomplex> S(S_k.data() + s*nk_*nao_*nao_ + k*nao_*nao_, nao_, nao_);
      CMMatrixX<dcomplex> F(F_k.data() + s*nk_*nao_*nao_ + k*nao_*nao_, nao_, nao_);

      for (int w = 0; w < nw_; ++w) {
        std::complex<double> muomega = wsample(w) + mu;
        G_w(w).matrix() = muomega * S - F - Sigma_w(w).matrix();
        G_w(w).matrix() = lusolver.compute(G_w(w).matrix()).inverse().eval();
      }
      MMatrixX<dcomplex>(G_t.data(), nts_, nao_ * nao_)
        = T_tw * MMatrixX<dcomplex>(G_w.data(), nw_, nao_ * nao_);
      for (int t = 0; t < nts_; ++t) {
        G_tskij(t, s, k).matrix() = G_t(t).matrix();
      }
    } // k
  } // s
}

} // symmetry_mbpt
