
#include "gw_block.h"

namespace symmetry_mbpt {

void gw_block_solver::compute_full_P0_t(const std::string &df_integral_file) {

  coul_int_ = new df_integral_block_t(df_integral_file, symm_utils_, prefix_);
  coul_int_->read_integrals();

  // intermediate objects
  tensor<dcomplex, 1> Y2_inP;
  tensor<dcomplex, 1> X1_tmQ;
  tensor<dcomplex, 1> X2_Ptm;

  P0_t_qQP_.reshape(nts_, symm_utils_.flat_aux_size());
  P0_t_qQP_.set_zero();

  for (size_t q = 0; q < nk_; ++q) {
    const auto &offsets_q = symm_utils_.aux_offsets(q);
    const auto &sizes_q = symm_utils_.aux_sizes(q);
    for (size_t k1 = 0; k1 < nk_; ++k1) {
      std::array<size_t, 4> k = symm_utils_.momentum_conservation({{k1, 0, q}});
      size_t k2 = k[3];  // k2 = k1 + q
      const auto &offsets_k1 = symm_utils_.ao_offsets(k1);
      const auto &sizes_k1 = symm_utils_.ao_sizes(k1);
      const auto &offsets_k2 = symm_utils_.ao_offsets(k2);
      const auto &sizes_k2 = symm_utils_.ao_sizes(k2);
      // take V as a flattened vector (Q, a, b)
      tensor<dcomplex, 1> V(symm_utils_.kpair_slice_sizes(k1, k2));
      coul_int_->symmetrize(V, k1, k2);
      const tensor<int, 2> &V_irreps = symm_utils_.V_irreps(k1, k2);
      const tensor<int, 1> &offsets_V = symm_utils_.V_offsets(k1, k2);

      for (int s = 0; s < ns_; ++s) {
        // loop over different blocks of aos follow irreps of V
        for (int ia = 0; ia < V_irreps.shape()[0]; ++ia) {
          // contraction at k1:
          int iq = V_irreps(ia, 0);
          int ik1 = V_irreps(ia, 1);
          int ik2 = V_irreps(ia, 2);
          size_t q_size = sizes_q(iq);
          size_t k1_size = sizes_k1(ik1);
          size_t k2_size = sizes_k2(ik2);
          int k1_offset = offsets_k1(ik1);
          int k2_offset = offsets_k2(ik2);
          int V_offset = offsets_V(ia);
          int Q_offset = offsets_q(iq);
          Y2_inP.reshape({k1_size*k2_size*q_size});
          X1_tmQ.reshape(Y2_inP.shape());
          // loop over time points
          for (int t = 0; t < nts_; ++t) {
            tensor_view<dcomplex, 1> G_tp = get_G(G_ts_kij_, nts_full_-1-t, s, k1);
            tensor_view<dcomplex, 1> G1_mn = get_G(G_ts_kij_, t, s, k2);
            tensor_view<dcomplex, 1> P0_QP = get_P(P0_t_qQP_, t, q);
            // transpose V_Qpm -> V_pmQ
            MMatrixX<dcomplex>(Y2_inP.data(), k1_size*k2_size, q_size)
              = MMatrixX<dcomplex>(V.data()+V_offset, q_size, k1_size*k2_size).transpose();
            // X1_t_mQ = G_t_p * V_pmQ; G_tp = G^{k1}(-t)_tp
            MMatrixX<dcomplex>(X1_tmQ.data(), k1_size, k2_size*q_size)
              = MMatrixX<dcomplex>(G_tp.data()+k1_offset, k1_size, k1_size)
                * MMatrixX<dcomplex>(Y2_inP.data(), k1_size, k2_size*q_size);

            // contraction at k2=k1+q:
            X2_Ptm.reshape(X1_tmQ.shape());
            // X2_Pt_m = (V_Pt_n)* G_m_n; G_mn = G^{k2}(t)_{mn}
            MMatrixX<dcomplex>(X2_Ptm.data(), q_size*k1_size, k2_size)
              = MMatrixX<dcomplex>(V.data()+V_offset, q_size*k1_size, k2_size).conjugate()
                * MMatrixX<dcomplex>(G1_mn.data()+k2_offset, k2_size, k2_size).transpose();

            // Pq0_QP = X2_Ptm X1_tmQ
            MMatrixX<dcomplex>(P0_QP.data()+Q_offset, q_size, q_size).transpose()
              += MMatrixX<dcomplex>(X2_Ptm.data(), q_size, k1_size*k2_size)
                * MMatrixX<dcomplex>(X1_tmQ.data(), k1_size*k2_size, q_size);
          } // V blocks
        } // t
      } // s
    } // k2
  } // k1
  P0_t_qQP_ *= -1.0 / nk_;
  if (ns_ == 1) { P0_t_qQP_ *= 2.0;}

  delete coul_int_;
}

void gw_block_solver::compute_Sigma_from_P_t(const std::string &df_integral_file, const tensor<dcomplex, 2> &P_t_qQP) {

  coul_int_ = new df_integral_block_t(df_integral_file, symm_utils_, prefix_);
  coul_int_->read_integrals();

  // intermediate objects
  tensor<dcomplex, 1> Y1_Qin;
  tensor<dcomplex, 1> Y2_inP;
  tensor<dcomplex, 1> V_nPj;

  Sigma_ts_kij_.set_zero();
  for (size_t q = 0; q < nk_; ++q) {
    const auto &offsets_q = symm_utils_.aux_offsets(q);
    const auto &sizes_q = symm_utils_.aux_sizes(q);
    for (size_t k1 = 0; k1 < nk_; ++k1) {
      std::array<size_t, 4> k = symm_utils_.momentum_conservation({{k1, q, 0}});
      size_t k2 = k[3]; // k2 = k1 - q
      const auto &offsets_k1 = symm_utils_.ao_offsets(k1);
      const auto &sizes_k1 = symm_utils_.ao_sizes(k1);
      const auto &offsets_k2 = symm_utils_.ao_offsets(k2);
      const auto &sizes_k2 = symm_utils_.ao_sizes(k2);
      // take V as a flattened vector (Q, a, b)
      tensor<dcomplex, 1> V(symm_utils_.kpair_slice_sizes(k1, k2));
      coul_int_->symmetrize(V, k1, k2);
      const tensor<int, 2> &V_irreps = symm_utils_.V_irreps(k1, k2);
      const tensor<int, 1> &offsets_V = symm_utils_.V_offsets(k1, k2);
      
      for (int s = 0; s < ns_; ++s) {
        for (int ia = 0; ia < V_irreps.shape()[0]; ++ia) {
          int iq = V_irreps(ia, 0);
          int ik1 = V_irreps(ia, 1);
          int ik2 = V_irreps(ia, 2);
          size_t q_size = sizes_q(iq);
          size_t k1_size = sizes_k1(ik1);
          size_t k2_size = sizes_k2(ik2);
          int k1_offset = offsets_k1(ik1);
          int k2_offset = offsets_k2(ik2);
          int V_offset = offsets_V(ia);
          int Q_offset = offsets_q(iq);
          Y2_inP.reshape({k1_size*k2_size*q_size});
          Y1_Qin.reshape(Y2_inP.shape());
          V_nPj.reshape({k1_size*k2_size*q_size});
          for (int t = 0; t < nts_; ++t) {
            Ctensor_view<dcomplex, 1> P_QP = get_const_P(P_t_qQP, t, q);
            tensor_view<dcomplex, 1> Sigma_ij = get_G(Sigma_ts_kij_, t, s, k1);
            tensor_view<dcomplex, 1> G1_mn = get_G(G_ts_kij_, t, s, k2);
            // contraction at k2 = k1 - q:
            // Y1_Qi,n = V_Qi,m * G1_m,n
            MMatrixX<dcomplex>(Y1_Qin.data(), q_size*k1_size, k2_size)
              = MMatrixX<dcomplex>(V.data()+V_offset, q_size*k1_size, k2_size)
                * MMatrixX<dcomplex>(G1_mn.data()+k2_offset, k2_size, k2_size);
            // contraction at q
            // Y2_in,P = Y1_Q,in * Pq_Q,P
            MMatrixX<dcomplex>(Y2_inP.data(), k1_size*k2_size, q_size)
              = MMatrixX<dcomplex>(Y1_Qin.data(), q_size, k1_size*k2_size).transpose()
                * CMMatrixX<dcomplex>(P_QP.data()+Q_offset, q_size, q_size);
            // transpose
            // V_k1k2_jn = V_k2k1_nj.conjugate()
            // V_Pjn -> V_nPj
            // Sigma_i,j = Y2_i,nP * VV_nP,j
            MMatrixX<dcomplex>(V_nPj.data(), k2_size, q_size*k1_size)
              = MMatrixX<dcomplex>(V.data()+V_offset, q_size*k1_size, k2_size).transpose().conjugate();
            MMatrixX<dcomplex>(Sigma_ij.data()+k1_offset, k1_size, k1_size)
              += MMatrixX<dcomplex>(Y2_inP.data(), k1_size, k2_size*q_size)
                * MMatrixX<dcomplex>(V_nPj.data(), k2_size*q_size, k1_size);
          }
        } // t
      } // s
    } // q
  } // k1
  Sigma_ts_kij_ *= -1.0 / nk_;
  delete coul_int_;
}

tensor_view<dcomplex, 1> gw_block_solver::get_G(tensor_view<dcomplex, 3> &G, int t, int s, int k) {
  return tensor_view<dcomplex, 1>(G(t, s).data()+ symm_utils_.kao_slice_offsets()(k), symm_utils_.kao_slice_sizes()(k));
}

tensor_view<dcomplex, 1> gw_block_solver::get_P(tensor<dcomplex, 2> &P, int t, int q) {
  return tensor_view<dcomplex, 1>(P(t).data()+ symm_utils_.qaux_slice_offsets()(q), symm_utils_.qaux_slice_sizes()(q));
}

Ctensor_view<dcomplex, 1> gw_block_solver::get_const_P(const tensor<dcomplex, 2> &P, int t, int q) {
  return Ctensor_view<dcomplex, 1>(P(t).data()+ symm_utils_.qaux_slice_offsets()(q), symm_utils_.qaux_slice_sizes()(q));
}

} // namespace symmetry_mbpt
