
#include "gw.h"

namespace symmetry_mbpt {

void gw_solver::solve(const std::string &df_integral_file,
                      const MMatrixX<dcomplex> &T_wt, const MMatrixX<dcomplex> &T_tw) {

  coul_int_ = new df_integral_t(df_integral_file, nao_, NQ_, symm_utils_, symm_, prefix_);
  tensor<dcomplex, 3> V(NQ_, nao_, nao_);

  tensor<dcomplex, 3> X1_tmQ(nao_, nao_, NQ_);
  tensor<dcomplex, 3> X2_Ptm(NQ_, nao_, nao_);
  tensor<dcomplex, 3> Y1_Qin(NQ_, nao_, nao_);
  tensor<dcomplex, 3> Y2_inP(nao_, nao_, NQ_);
  tensor<dcomplex, 3> V_nPj(nao_, NQ_, nao_);

  Sigma_tskij_.set_zero();

  for (size_t q = 0; q < nk_; ++q) {
    // compute P0
    P0_tQP_.set_zero();
    for (size_t k1 = 0; k1 < nk_; ++k1) {
      std::array<size_t, 4> k = symm_utils_.momentum_conservation({{k1, 0, q}});
      size_t k2 = k[3];  // k2 = k1 + q
      coul_int_->read_integrals(k1, k2);
      coul_int_->symmetrize(V, k1, k2);
      for (int t = 0; t < nts_; ++t) {
        MMatrixX<dcomplex> P0_QP(P0_tQP_.data() + t * NQ_ * NQ_, NQ_, NQ_);
        for (int s = 0; s < ns_; ++s) {
          MMatrixX<dcomplex> G_tp = get_G(G_tskij_, nts_full_-1-t, s, k1);
          MMatrixX<dcomplex> G1_mn = get_G(G_tskij_, t, s, k2);
          P0_contraction(P0_QP, G_tp, G1_mn, V, X1_tmQ, X2_Ptm, Y2_inP, nao_, NQ_);
        } // s
      } // t
    } // k1
    P0_tQP_ *= -1.0 / nk_;
    if (ns_ == 1) { P0_tQP_ *= 2.0;}

    // compute P from P0
    eval_P_from_P0(P0_tQP_, T_wt, T_tw, P0_wQP_, nw_b_, nts_, NQ_);

    // accumulate results to Sigma
    for (size_t k1 = 0; k1 < nk_; ++k1) {
      std::array<size_t, 4> k = symm_utils_.momentum_conservation({{k1, q, 0}});
      size_t k2 = k[3]; // k2 = k1 - q
      coul_int_->read_integrals(k1, k2);
      coul_int_->symmetrize(V, k1, k2);
      for (int t = 0; t < nts_; ++t) {
        const CMMatrixX<dcomplex> P0_QP(P0_tQP_.data() + t * NQ_ * NQ_, NQ_, NQ_);
        for (int s = 0; s < ns_; ++s) {
          MMatrixX<dcomplex> Sigma_ij = get_G(Sigma_tskij_, t, s, k1);
          const MMatrixX<dcomplex> G1_mn = get_G(G_tskij_, t, s, k2);
          Sigma_contraction(Sigma_ij, G1_mn, P0_QP, V, V_nPj, Y1_Qin, Y2_inP, nao_, NQ_);
        } // s
      } // t
    } // k1

  } // q
  Sigma_tskij_ *= -1.0 / nk_;

  delete coul_int_;
}

void gw_solver::compute_full_P0_t(const std::string &df_integral_file) {

  coul_int_ = new df_integral_t(df_integral_file, nao_, NQ_, symm_utils_, symm_, prefix_);
  tensor<dcomplex, 3> V(NQ_, nao_, nao_);

  tensor<dcomplex, 3> X1_tmQ(nao_, nao_, NQ_);
  tensor<dcomplex, 3> X2_Ptm(NQ_, nao_, nao_);
  tensor<dcomplex, 3> Y2_inP(nao_, nao_, NQ_);

  P0_tqQP_.reshape(nts_, nk_, NQ_, NQ_);
  P0_tqQP_.set_zero();

  for (size_t q = 0; q < nk_; ++q) {
    for (size_t k1 = 0; k1 < nk_; ++k1) {
      std::array<size_t, 4> k = symm_utils_.momentum_conservation({{k1, 0, q}});
      size_t k2 = k[3];  // k2 = k1 + q
      coul_int_->read_integrals(k1, k2);
      coul_int_->symmetrize(V, k1, k2);
      for (int t = 0; t < nts_; ++t) {
        MMatrixX<dcomplex> P0_QP(P0_tqQP_.data() + t * nk_ * NQ_ * NQ_ + q * NQ_ * NQ_, NQ_, NQ_);
        for (int s = 0; s < ns_; ++s) {
          MMatrixX<dcomplex> G_tp = get_G(G_tskij_, nts_full_-1-t, s, k1);
          MMatrixX<dcomplex> G1_mn = get_G(G_tskij_, t, s, k2);
          P0_contraction(P0_QP, G_tp, G1_mn, V, X1_tmQ, X2_Ptm, Y2_inP, nao_, NQ_);
        } // s
      } // t
    } // k1
  }
  P0_tqQP_ *= -1.0 / nk_;
  if (ns_ == 1) { P0_tqQP_ *= 2.0;}
  delete coul_int_;
}

void gw_solver::compute_Sigma_from_P_t(const std::string &df_integral_file, const tensor<dcomplex, 4> &P_tqQP) {

  coul_int_ = new df_integral_t(df_integral_file, nao_, NQ_, symm_utils_, symm_, prefix_);
  tensor<dcomplex, 3> V(NQ_, nao_, nao_);

  tensor<dcomplex, 3> Y1_Qin(NQ_, nao_, nao_);
  tensor<dcomplex, 3> Y2_inP(nao_, nao_, NQ_);
  tensor<dcomplex, 3> V_nPj(nao_, NQ_, nao_);

  Sigma_tskij_.set_zero();

  for (size_t q = 0; q < nk_; ++q) {
    for (size_t k1 = 0; k1 < nk_; ++k1) {
      std::array<size_t, 4> k = symm_utils_.momentum_conservation({{k1, q, 0}});
      size_t k2 = k[3]; // k2 = k1 - q
      coul_int_->read_integrals(k1, k2);
      coul_int_->symmetrize(V, k1, k2);
      for (int t = 0; t < nts_; ++t) {
        const CMMatrixX<dcomplex> P_QP(P_tqQP.data() + t * nk_ * NQ_ * NQ_ + q * NQ_ * NQ_, NQ_, NQ_);
        for (int s = 0; s < ns_; ++s) {
          MMatrixX<dcomplex> Sigma_ij = get_G(Sigma_tskij_, t, s, k1);
          const MMatrixX<dcomplex> G1_mn = get_G(G_tskij_, t, s, k2);
          Sigma_contraction(Sigma_ij, G1_mn, P_QP, V, V_nPj, Y1_Qin, Y2_inP, nao_, NQ_);
        } // s
      } // t
    } // k1
  } // q
  Sigma_tskij_ *= -1.0 / nk_;
  delete coul_int_;
}

void gw_solver::compute_P_from_P0(const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw) {

  tensor<dcomplex, 4> P0_wqQP(nw_b_, nk_, NQ_, NQ_);
  // Solve Dyson-like eqn for nw_b frequency points
  MatrixXcd identity = MatrixXcd::Identity(NQ_, NQ_);
  //Eigen::FullPivLU<MatrixXcd> lusolver(_NQ,_NQ);
  Eigen::LDLT<MatrixXcd> ldltsolver(NQ_);

  for (size_t it = 0; it < nts_full_ / 2; ++it) {
    for (int q = 0; q < nk_; ++q) {
      P0_tqQP_(nts_full_-it-1, q).matrix() = P0_tqQP_(it, q).matrix();
    }
  }
  make_hermition(P0_tqQP_);
  MMatrixX<dcomplex>(P0_wqQP.data(), nw_b_, nk_*NQ_*NQ_)
      = T_wt * MMatrixX<dcomplex>(P0_tqQP_.data()+nk_*NQ_*NQ_, nts_full_-2, nk_*NQ_*NQ_);

  // Solve Dyson-like eqn for ncheb frequency points
  Matrix<dcomplex> temp(NQ_, NQ_);
  for (size_t n = 0; n < nw_b_; ++n) {
    for (int q = 0; q < nk_; ++q) {
      temp = identity - MMatrixX<dcomplex>(P0_wqQP(n, q).data(), NQ_, NQ_);
      temp = ldltsolver.compute(temp).solve(identity).eval();
      temp = 0.5 * (temp + temp.conjugate().transpose().eval());
      MMatrixX<dcomplex>(P0_wqQP(n, q).data(), NQ_, NQ_)
          = (temp * MMatrixX<dcomplex>(P0_wqQP(n, q).data(), NQ_, NQ_)).eval();
    }
  }
  MMatrixX<dcomplex>(P0_tqQP_.data(), nts_full_, nk_ * NQ_ * NQ_)
      = T_tw * MMatrixX<dcomplex>(P0_wqQP.data(), nw_b_, nk_ * NQ_ * NQ_);
  make_hermition(P0_tqQP_);
}

MMatrixX<dcomplex> gw_solver::get_G(tensor_view<dcomplex, 5> &G, int t, int s, int k) const {
  return MMatrixX<dcomplex>(G.data() + t * ns_ * nk_ * nao_ * nao_
                                     + s * nk_ * nao_ * nao_ +
                                     + k * nao_ * nao_, nao_, nao_);
}

template<typename T>
void gw_solver::P0_contraction(MMatrixX<T> &Pq0_QP, const MMatrixX<T> &G_tp, const MMatrixX<T> &G1_mn,
                               const tensor<T, 3> &V_Pqm,
                               tensor<T, 3> &X1_tmQ, tensor<T, 3> &X2_Ptm, tensor<T, 3> &Y_inP, int nao, int NQ) {

  // first contraction
  MMatrixX<T>(Y_inP.data(), nao * nao, NQ) = CMMatrixX<T>(V_Pqm.data(), NQ, nao * nao).transpose();
  MMatrixX<T>(X1_tmQ.data(), nao, nao * NQ) = G_tp * MMatrixX<T>(Y_inP.data(), nao, nao * NQ);
  // second contraction
  MMatrixX<T>(X2_Ptm.data(), NQ * nao, nao)
    = CMMatrixX<T>(V_Pqm.data(), NQ * nao, nao).conjugate() * G1_mn.transpose();
  // third contraction
  Pq0_QP.transpose() += MMatrixX<T>(X2_Ptm.data(), NQ, nao * nao) * MMatrixX<T>(X1_tmQ.data(), nao * nao, NQ);
}

template<typename T>
void gw_solver::Sigma_contraction(MMatrixX<T> &Sigma_ij,
                                  const MMatrixX<T> &G1_mn, const CMMatrixX<T> &P_QP, const tensor<T, 3> &V_Qim,
                                  tensor<T, 3> &V_nPj, tensor<T, 3> &Y1_Qin, tensor<T, 3> &Y2_inP, int nao, int NQ) {

  // first contraction
  MMatrixX<T>(Y1_Qin.data(), NQ * nao, nao) = CMMatrixX<T>(V_Qim.data(), NQ * nao, nao) * G1_mn;
  // second contraction
  MMatrixX<T>(Y2_inP.data(), nao * nao, NQ) = MMatrixX<T>(Y1_Qin.data(), NQ, nao * nao).transpose() * P_QP;
  // third contraction
  MMatrixX<T>(V_nPj.data(), nao, NQ * nao) = CMMatrixX<T>(V_Qim.data(), NQ * nao, nao).transpose().conjugate();
  Sigma_ij += MMatrixX<T>(Y2_inP.data(), nao, nao * NQ) * MMatrixX<T>(V_nPj.data(), nao * NQ, nao);
}

template<typename T>
void gw_solver::eval_P_from_P0(tensor<T, 4> &P0_tQP,
                               const MMatrixX<T> &T_wt, const MMatrixX<T> &T_tw,
                               tensor<T, 4> &P0_wQP, int nw, int nt, int NQ) {
  MMatrixX<T>(P0_wQP.data(), nw, NQ * NQ) = T_wt * CMMatrixX<T>(P0_tQP.data(), nt, NQ * NQ);

  // Solve Dyson-like eqn for ncheb frequency points
  Matrix<T> identity = Matrix<T>::Identity(NQ, NQ);
  Eigen::LDLT<Matrix<T>> ldltsolver(NQ);
  Matrix<T> temp(NQ, NQ);
  for (size_t n = 0; n < nw; ++n) {
    temp = identity - MMatrixX<T>(P0_wQP.data()+n*NQ*NQ, NQ, NQ);
    temp = ldltsolver.compute(temp).solve(identity).eval();
    temp = 0.5 * (temp + temp.conjugate().transpose().eval());
    MMatrixX<T>(P0_wQP.data()+n*NQ*NQ, NQ, NQ) = (temp * MMatrixX<T>(P0_wQP.data()+n*NQ*NQ, NQ, NQ)).eval();
  }
  MMatrixX<T>(P0_tQP.data(), nt, NQ * NQ) = T_tw * MMatrixX<T>(P0_wQP.data(), nw, NQ * NQ);
}

template void gw_solver::P0_contraction<dcomplex>(MMatrixX<dcomplex> &Pq0_QP, const MMatrixX<dcomplex> &G_tp,
                                                  const MMatrixX<dcomplex> &G1_mn, const tensor<dcomplex, 3> &V_Pqm,
                                                  tensor<dcomplex, 3> &X1_tmQ, tensor<dcomplex, 3> &X2_Ptm,
                                                  tensor<dcomplex, 3> &Y_inP, int nao, int NQ);

template void gw_solver::Sigma_contraction<dcomplex>(MMatrixX<dcomplex> &Sigma_ij,
                                                     const MMatrixX<dcomplex> &G1_mn, const CMMatrixX<dcomplex> &P_QP,
                                                     const tensor<dcomplex, 3> &V_Qim, tensor<dcomplex, 3> &V_nPj,
                                                     tensor<dcomplex, 3> &Y1_Qin, tensor<dcomplex, 3> &Y2_inP,
                                                     int nao, int NQ);

template void gw_solver::eval_P_from_P0<dcomplex>(tensor<dcomplex, 4> &P_tQP,
                                                  const MMatrixX<dcomplex> &T_wt, const MMatrixX<dcomplex> &T_tw,
                                                  tensor<dcomplex, 4> &P0_wQP, int nw, int nt, int NQ);

} // namespace symmetry_mbpt