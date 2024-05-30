#include "analysis.h"
#include "common.h"

namespace symmetry_mbpt {

  double HF_complexity_estimation_full(int world_rank, const symmetry_utils_t &symm_utils, int ns, int ink,
                                       matmul_type mat_type) {
    int nk = symm_utils.nk();
    if (ink == 0) ink = nk;

    int nao = symm_utils.nao();
    int NQ = symm_utils.NQ();
    double naosq = nao * nao;
    // Direct diagram
    double flop_count_direct =
        nk*ns*matmul_cost(1, NQ, naosq, mat_type) + ink*ns*matmul_cost(1, naosq, NQ, mat_type);
    // Exchange diagram
    double flop_count_exchange =
        ink*ns*nk*(matmul_cost(NQ*nao, nao, nao, mat_type) + matmul_cost(nao, nao, NQ*nao, mat_type));
    // dm rotation
    double flop_count_rotate = 0.;
    if (ink < nk) {
      for (int k = 0; k < nk; ++k) {
        int ck = symm_utils.conj_index()[k];
        if (symm_utils.index()[ck] == ck) continue;
        else flop_count_rotate += 2*ns*matmul_cost(nao, nao, nao);
      }
    }

    double hf_total_flops = flop_count_direct + flop_count_exchange + flop_count_rotate;

    if (!world_rank) {
      std::cout << "############ Total HF Operations per Iteration ############" << std::endl;
      std::cout << "Total:                          " << hf_total_flops << std::endl;
      std::cout << "Matmul (Direct diagram):        " << flop_count_direct << std::endl;
      std::cout << "Matmul (Exchange diagram):      " << flop_count_exchange << std::endl;
      std::cout << "Matmul (Density matrix rotate): " << flop_count_rotate << std::endl;
      std::cout << "###########################################################" << std::endl;
    }

    return hf_total_flops;
  }

  double HF_complexity_estimation_block(int world_rank, const symmetry_utils_t &symm_utils, int ns, int ink,
                                        matmul_type mat_type) {
    int nk = symm_utils.nk();
    if (ink == 0) ink = nk;

    /*
     * dm rotate
     */
    double flop_count_rotate = 0.;
    if (ink < nk) {
      for (int k = 0; k < nk; ++k) {
        int ck = symm_utils.conj_index()[k];
        if (symm_utils.index()[ck] == ck) continue;
        int ick = symm_utils.irre_index()[ck];
        const auto &sizes_k = symm_utils.ao_sizes(ck);
        const auto &sizes_ik = symm_utils.ao_sizes_irre(ick);
        const auto &irreps = symm_utils.orep_irreps()[ck];
        for (int ia = 0; ia < irreps.shape()[0]; ++ia) {
          int ia1 = irreps(ia, 0);
          int ia2 = irreps(ia, 1);
          int k_size = sizes_k(ia1);
          int ik_size = sizes_ik(ia2);
          flop_count_rotate += ns * (matmul_cost(k_size, ik_size, ik_size) + matmul_cost(k_size, k_size, ik_size));
        }
      }
    }

    /*
     * Direct diagram
     */
    double flop_count_direct = 0.;
    int q = 0;
    const auto &sizes_q0 = symm_utils.aux_sizes(q);
    // TODO: check if could/should simplify this to ink
    // first contraction
    for (int ikp = 0; ikp < nk; ++ikp) {
      int kp_ir = ikp;
      const tensor<int, 2> &V_irreps = symm_utils.V_irreps(kp_ir, kp_ir);
      int n_block = V_irreps.shape()[0];
      const auto &sizes_k = symm_utils.ao_sizes(ikp);
      for (int ia = 0; ia < n_block; ++ia) {
        int iq = V_irreps(ia, 0);
        int ik1 = V_irreps(ia, 1);
        int ik2 = V_irreps(ia, 2);
        // dm is block diagonal
        if (ik1 == ik2) {
          flop_count_direct += ns*matmul_cost(1, sizes_q0(iq), sizes_k(ik1)*sizes_k(ik1), mat_type);
        }
      }
    } // ikp
    // second contraction
    for (int ik = 0; ik < ink; ++ik) {
      int k_ir = symm_utils.irre_list()[ik];
      const tensor<int, 2> &V_irreps = symm_utils.V_irreps(k_ir, k_ir);
      int n_block = V_irreps.shape()[0];
      const auto &sizes_k = symm_utils.ao_sizes(k_ir);
      for (int ia = 0; ia < n_block; ++ia) {
        int iq = V_irreps(ia, 0);
        int ik1 = V_irreps(ia, 1);
        int ik2 = V_irreps(ia, 2);
        if (ik1 == ik2) {
          flop_count_direct += ns*matmul_cost(1, sizes_k(ik1)*sizes_k(ik1), sizes_q0(iq), mat_type);
        }
      }
    }

    /*
     * Exchange diagram
     */
    double flop_count_exchange = 0.;
    for (int ik = 0; ik < ink; ++ik) {
      size_t k1 = symm_utils.irre_list()[ik];
      for (size_t k2 = 0; k2 < nk; ++k2) {
        std::array<size_t, 4> k_list = symm_utils.momentum_conservation({{k2, k1, 0}});
        q = k_list[3]; // k2+q=k1
        const auto &sizes_q = symm_utils.aux_sizes(q);
        const auto &sizes_k1 = symm_utils.ao_sizes(k1);
        const auto &sizes_k2 = symm_utils.ao_sizes(k2);
        const tensor<int, 2> &V_irreps = symm_utils.V_irreps(k1, k2);
        int n_block = V_irreps.shape()[0];
        for (int ia = 0; ia < n_block; ++ia) {
          int iq = V_irreps(ia, 0);
          int ik1 = V_irreps(ia, 1);
          int ik2 = V_irreps(ia, 2);
          size_t q_size = sizes_q(iq);
          size_t k1_size = sizes_k1(ik1);
          size_t k2_size = sizes_k2(ik2);
          flop_count_exchange +=
              ns*(matmul_cost(q_size*k1_size, k2_size, k2_size, mat_type)
                  + matmul_cost(k1_size, k1_size, k2_size*q_size, mat_type));
        }
      } // k2
    } // k1

    double hf_total_flops = flop_count_direct + flop_count_exchange + flop_count_rotate;

    if (!world_rank) {
      std::cout << "############ Total HF Operations per Iteration with Block ############" << std::endl;
      std::cout << "Total:                          " << hf_total_flops << std::endl;
      std::cout << "Matmul (Direct diagram):        " << flop_count_direct << std::endl;
      std::cout << "Matmul (Exchange diagram):      " << flop_count_exchange << std::endl;
      std::cout << "Matmul (Density matrix rotate): " << flop_count_rotate << std::endl;
      std::cout << "######################################################################" << std::endl;
    }

    return hf_total_flops;
  }

  double GW_complexity_estimation_full(int world_rank, const symmetry_utils_t &symm_utils,
                                       int ns, int nts, int nw_b, int ink, matmul_type mat_type) {
    int nk = symm_utils.nk();
    if (ink == 0) ink = nk;

    int nao = symm_utils.nao();
    int NQ = symm_utils.NQ();
    double NQsq = NQ * NQ;
    double naosq = nao * nao;

    //first set of matmuls (P0 contraction)
    double flop_count_firstmatmul = ink*nk*ns*nts/2.*(
        matmul_cost(nao*NQ, nao, nao, mat_type)  // X1_t_mQ = G_t_p * V_pmQ;
        + matmul_cost(NQ*nao, nao, nao, mat_type) // X2_Pt_m = (V_Pt_n)* * G_m_n;
        + matmul_cost(NQ, NQ, naosq, mat_type)    // Pq0_QP=X2_Ptm Q1_tmQ
    );

    //Fourier transform forward and back
    double flop_count_fourier = ink*(matmul_cost(NQsq, nts, nw_b) + matmul_cost(NQsq, nw_b, nts));
    //approximate LU and backsubst cost (note we are doing cholesky which is cheaper)
    double flop_count_solver = 2./3.*ink*nw_b*(NQsq*NQ + NQsq);

    //second set of matmuls
    double flop_count_secondmatmul = ink*nk*ns*nts*(
        matmul_cost(NQ*nao, nao, nao, mat_type)  // Y1_Qin = V_Qim * G1_mn;
        + matmul_cost(naosq, NQ, NQ, mat_type)    // Y2_inP = Y1_Qin * Pq_QP
        + matmul_cost(nao, nao, NQ*nao, mat_type) // Sigma_ij = Y2_inP V_nPj
    );

    double flop_count_G_rotate = 0.;
    double flop_count_P_rotate = 0.;
    if (ink < nk) {
      for (int k = 0; k < nk; ++k) {
        int ck = symm_utils.conj_index()[k];
        if (symm_utils.index()[ck] == ck) continue;
        else {
          flop_count_G_rotate += 2*nts*ns*matmul_cost(nao, nao, nao);
          flop_count_P_rotate += 2*nts*matmul_cost(NQ, NQ, NQ);
        }
      }
    }

    double flop_count = flop_count_firstmatmul + flop_count_fourier + flop_count_solver + flop_count_secondmatmul
        + flop_count_G_rotate + flop_count_P_rotate;

    if (!world_rank) {
      std::cout << "############ Total GW Operations per Iteration ############" << std::endl;
      std::cout << "Total:         " << flop_count << std::endl;
      std::cout << "First matmul:  " << flop_count_firstmatmul << std::endl;
      std::cout << "Fourier:       " << flop_count_fourier << std::endl;
      std::cout << "Solver:        " << flop_count_solver << std::endl;
      std::cout << "Second matmul: " << flop_count_secondmatmul << std::endl;
      std::cout << "G rotate:      " << flop_count_G_rotate << std::endl;
      std::cout << "P rotate:      " << flop_count_P_rotate << std::endl;
      std::cout << "###########################################################" << std::endl;
    }

    return flop_count;
  }

  double GW_complexity_estimation_block(int world_rank,
                                        const symmetry_utils_t &symm_utils, int ns, int nts, int nw_b, int ink,
                                        matmul_type mat_type) {
    int nk = symm_utils.nk();
    if (ink == 0) ink = nk;
    
    tensor<int, 2> qk_pairs(ink*nk, 3);
    for (int i = 0; i < ink; ++i) {
      for (int j = 0; j < nk; ++j) {
        int index = i * nk + j;
        qk_pairs(index, 0) = i;
        qk_pairs(index, 1) = symm_utils.irre_list()[i];
        qk_pairs(index, 2) = j;
      }
    }

    int n_qkpair = qk_pairs.shape()[0];
    double flop_count_firstmatmul = 0.;
    for (int i = 0; i < n_qkpair; ++i) {
      size_t q_iBZ = qk_pairs(i, 0);  // index in iBZ
      size_t q = qk_pairs(i, 1); // index in full BZ
      size_t k1 = qk_pairs(i, 2); // index in full BZ
      std::array<size_t, 4> k_vector = symm_utils.momentum_conservation({{k1, 0, q}});
      size_t k2 = k_vector[3]; // k2 = k1 + q
      const auto &sizes_q = symm_utils.aux_sizes(q);
      const auto &sizes_k1 = symm_utils.ao_sizes(k1);
      const auto &sizes_k2 = symm_utils.ao_sizes(k2);

      const tensor<int, 2> &V_irreps = symm_utils.V_irreps(k1, k2);
      for (int ia = 0; ia < V_irreps.shape()[0]; ++ia) {
        // contraction at k1:
        int iq = V_irreps(ia, 0);
        int ik1 = V_irreps(ia, 1);
        int ik2 = V_irreps(ia, 2);
        size_t q_size = sizes_q(iq);
        size_t k1_size = sizes_k1(ik1);
        size_t k2_size = sizes_k2(ik2);
        flop_count_firstmatmul += ns*nts/2.*(
            matmul_cost(k2_size*q_size, k1_size, k1_size, mat_type)
            + matmul_cost(q_size*k1_size, k2_size, k2_size, mat_type)
            + matmul_cost(q_size, q_size, k1_size*k2_size, mat_type)
            );
      }
    }

    double flop_count_fourier = 0.;
    double flop_count_solver = 0.;
    for (int q = 0; q < ink; ++q) {
      int q_slice_size = symm_utils.qaux_slice_sizes_irre()(q);
      flop_count_fourier += matmul_cost(q_slice_size, nts, nw_b) + matmul_cost(q_slice_size, nw_b, nts);

      int n_block = symm_utils.aux_sizes_irre(q).size();
      const auto &sizes_q = symm_utils.aux_sizes_irre(q);
      for (int ia = 0; ia < n_block; ++ia) {
        int q_size = sizes_q(ia);
        flop_count_solver += 2./3.*nw_b*(q_size*q_size*q_size + q_size*q_size);
      }
    }

    double flop_count_secondmatmul = 0.;
    for (int i = 0; i < n_qkpair; ++i) {
      size_t k1_iBZ = qk_pairs(i, 0);  // index in iBZ
      size_t k1 = qk_pairs(i, 1); // index in full BZ
      size_t q = qk_pairs(i, 2); // index in full BZ
      size_t q_iBZ = symm_utils.irre_index()[q]; // index in iBZ for corresponding iBZ(q)
      std::array<size_t, 4> k_vector = symm_utils.momentum_conservation({{k1, q, 0}});
      size_t k2 = k_vector[3]; // k2 = k1 - q
      const auto &sizes_q = symm_utils.aux_sizes(q);
      const auto &sizes_k1 = symm_utils.ao_sizes(k1);
      const auto &sizes_k2 = symm_utils.ao_sizes(k2);

      const tensor<int, 2> &V_irreps = symm_utils.V_irreps(k1, k2);
      for (int ia = 0; ia < V_irreps.shape()[0]; ++ia) {
        // contraction at k1:
        int iq = V_irreps(ia, 0);
        int ik1 = V_irreps(ia, 1);
        int ik2 = V_irreps(ia, 2);
        size_t q_size = sizes_q(iq);
        size_t k1_size = sizes_k1(ik1);
        size_t k2_size = sizes_k2(ik2);
        flop_count_secondmatmul += ns*nts*(
            matmul_cost(q_size*k1_size, k2_size, k2_size, mat_type)
            + matmul_cost(k1_size*k2_size, q_size, q_size, mat_type)
            + matmul_cost(k1_size, k1_size, k2_size*q_size, mat_type)
            );
      }
    }

    double flop_count_G_rotate = 0.;
    double flop_count_P_rotate = 0.;
    if (ink < nk) {
      for (int k = 0; k < nk; ++k) {
        int ck = symm_utils.conj_index()[k];
        if (symm_utils.index()[ck] == ck) continue;
        int ick = symm_utils.irre_index()[ck];

        const auto &sizes_k = symm_utils.ao_sizes(ck);
        const auto &sizes_ik = symm_utils.ao_sizes_irre(ick);
        const auto &irreps = symm_utils.orep_irreps()[ck];
        for (int ia = 0; ia < irreps.shape()[0]; ++ia) {
          int ia1 = irreps(ia, 0);
          int ia2 = irreps(ia, 1);
          int k_size = sizes_k(ia1);
          int ik_size = sizes_ik(ia2);
          flop_count_G_rotate
              += nts * ns * (matmul_cost(k_size, ik_size, ik_size) + matmul_cost(k_size, k_size, ik_size));
        }

        const auto &sizes_q = symm_utils.aux_sizes(ck);
        const auto &sizes_iq = symm_utils.aux_sizes_irre(ick);
        const auto &irreps_aux = symm_utils.auxrep_irreps()[ck];
        for (int ia = 0; ia < irreps_aux.shape()[0]; ++ia) {
          int ia1 = irreps_aux(ia, 0);
          int ia2 = irreps_aux(ia, 1);
          int q_size = sizes_q(ia1);
          int iq_size = sizes_iq(ia2);
          flop_count_P_rotate
              += nts * ns * (matmul_cost(q_size, iq_size, iq_size) + matmul_cost(q_size, q_size, iq_size));
        }
      }
    }

    double flop_count = flop_count_firstmatmul + flop_count_fourier + flop_count_solver + flop_count_secondmatmul
        + flop_count_G_rotate + flop_count_P_rotate;

    if (!world_rank) {
      std::cout << "############ Total GW Operations per Iteration with Block ############" << std::endl;
      std::cout << "Total:         " << flop_count << std::endl;
      std::cout << "First matmul:  " << flop_count_firstmatmul << std::endl;
      std::cout << "Fourier:       " << flop_count_fourier << std::endl;
      std::cout << "Solver:        " << flop_count_solver << std::endl;
      std::cout << "Second matmul: " << flop_count_secondmatmul << std::endl;
      std::cout << "G rotate:      " << flop_count_G_rotate << std::endl;
      std::cout << "P rotate:      " << flop_count_P_rotate << std::endl;
      std::cout << "######################################################################" << std::endl;
    }

    return flop_count;
  }

} // namespace symmetry_mbpt
