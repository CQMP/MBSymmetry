#include "gtest/gtest.h"

#include "dyson_block.h"
#include "gw_block.h"
#include "block_utils.h"
#include "gw.h"

#include "sc_loop_block_t.h"

TEST(CuGWP0, LiHnk2) {

  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  params_t p = symmetry_mbpt::default_parameters();
  p.IR = true;
  p.dfintegral_file = input_block_path + std::string("/df_int");
  p.dfintegral_hf_file = input_block_path + std::string("/df_int");
  p.input_file = input_block_path + std::string("/input_flat.h5");
  p.symmetry_file = input_block_path + std::string("/symmetry_info.h5");
  p.tnc_f_file = ir_path + "/1e3_72.h5";
  p.tnc_b_file = ir_path + "/1e3_72.h5";
  p.ns = 2;
  p.ni = 72;
  p.nk = 8;
  p.beta = 5;
  p.ink = 8;
  p.mu = 0.0;
  p.scf_type = cuGW;
  p.nt_batch = 4;

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();
  sc.solver().solve_P0();

  /*
   * Block ref
   */
  if (!sc.world_rank()) {
    std::cout << "computing P0 reference results" << std::endl;
    symmetry_utils_t symm_utils(p);

    int nts_full = p.ni+2;
    int ns = p.ns;
    int nts = 10;
    std::string df_path = p.dfintegral_file;

    tensor<dcomplex, 3> G_block(nts_full, ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Gb_ts_kij(G_block.data(), nts_full, ns, symm_utils.flat_ao_size());

    tensor<dcomplex, 3> Sigma_full_block(nts_full, ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Sigmafb_ts_kij(Sigma_full_block.data(), nts_full, ns, symm_utils.flat_ao_size());
    Sigmafb_ts_kij.set_zero();

    transformer_t transformer(p.ni, p.beta, p.tnc_f_file, p.tnc_b_file);
    const MatrixX<dcomplex> &T_wt = transformer.Tnt();
    const MatrixX<dcomplex> &T_tw = transformer.Ttn();
    column<dcomplex> w_fermi(p.ni);
    for (int i = 0; i < p.ni; ++i) {
      w_fermi(i) = transformer.omega(transformer.wsample_fermi()[i], 1);
    }

    dyson_block gb_solver(symm_utils, ns, nts_full, transformer.nw());
    gb_solver.read_input(p.input_file);
    gb_solver.solve(Gb_ts_kij, Sigmafb_ts_kij, 0.0, T_wt, T_tw, w_fermi);

    tensor<dcomplex, 3> G_diff = Gb_ts_kij - sc.G_tau();
    EXPECT_LE(Mcolumn<dcomplex>(G_diff.data(), G_diff.size()).cwiseAbs().maxCoeff(), 1e-10);

    tensor<dcomplex, 3> Sigma_block(nts, ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Sigmab_ts_kij(Sigma_block.data(), nts, ns, symm_utils.flat_ao_size());

    gw_block_solver gw_block(symm_utils, Gb_ts_kij, Sigmab_ts_kij, nts, nts_full, ns, transformer.nw_b(), "flat_");

    gw_block.compute_full_P0_t(df_path);
    //gw_block.compute_Sigma_from_P_t(df_path, gw_block.P0_t_qQP());
    std::cout << "compute reference P0 done" << std::endl;

    /*auto P0_diff = CMcolumn<dcomplex>(sc.solver().P0_q_t_QP().data(), nts*symm_utils.flat_aux_size())
                   - CMcolumn<dcomplex>(gw_block.P0_t_qQP().data(), nts*symm_utils.flat_aux_size());
    EXPECT_LE(P0_diff.cwiseAbs().maxCoeff(), 1e-10);*/

    for (int q = 0; q < p.nk; ++q) {
      int qslice_size = symm_utils.qaux_slice_sizes()(q);
      int qslice_offset = symm_utils.qaux_slice_offsets()(q);
      for (int t = 0; t < nts; ++t) {
        CMcolumn<dcomplex> P0_compute(sc.solver().P0_q_t_QP().data() + nts_full*qslice_offset + t*qslice_size,
                                      qslice_size);
        CMcolumn<dcomplex> P0_ref(gw_block.P0_t_qQP()(t).data()+qslice_offset, qslice_size);
        auto P0_diff = P0_compute - P0_ref;
        //std::cout << CMcolumn<dcomplex>(
            //gw_block.P0_t_qQP()(t).data(), symm_utils.flat_aux_size()).cwiseAbs().maxCoeff() << std::endl;
        //std::cout << P0_diff.cwiseAbs().maxCoeff() << std::endl;
        EXPECT_LE(P0_diff.cwiseAbs().maxCoeff(), 1e-10);
      }
    }
  }
}

TEST(CuGWSigma, LiHnk2) {

  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  params_t p = symmetry_mbpt::default_parameters();
  p.IR = true;
  p.dfintegral_file = input_block_path + std::string("/df_int");
  p.dfintegral_hf_file = input_block_path + std::string("/df_int");
  p.input_file = input_block_path + std::string("/input_flat.h5");
  p.symmetry_file = input_block_path + std::string("/symmetry_info.h5");
  p.tnc_f_file = ir_path + "/1e3_72.h5";
  p.tnc_b_file = ir_path + "/1e3_72.h5";
  p.ns = 2;
  p.ni = 72;
  p.nk = 8;
  p.beta = 5;
  p.ink = 8;
  p.mu = 0.0;
  p.scf_type = cuGW;
  p.nt_batch = 4;

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();
  //sc.solver().solve();
  sc.solver().solve_sigma();

  /*
   * Block ref
   */
  if (!sc.world_rank()) {
    std::cout << "computing Sigma reference results" << std::endl;
    symmetry_utils_t symm_utils(p);

    int nts_full = p.ni+2;
    int ns = p.ns;
    int nts = 5;
    std::string df_path = p.dfintegral_file;

    tensor<dcomplex, 3> G_block(nts_full, ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Gb_ts_kij(G_block.data(), nts_full, ns, symm_utils.flat_ao_size());

    tensor<dcomplex, 3> Sigma_full_block(nts_full, ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Sigmafb_ts_kij(Sigma_full_block.data(), nts_full, ns, symm_utils.flat_ao_size());
    Sigmafb_ts_kij.set_zero();

    transformer_t transformer(p.ni, p.beta, p.tnc_f_file, p.tnc_b_file);
    const MatrixX<dcomplex> &T_wt = transformer.Tnt();
    const MatrixX<dcomplex> &T_tw = transformer.Ttn();
    column<dcomplex> w_fermi(p.ni);
    for (int i = 0; i < p.ni; ++i) {
      w_fermi(i) = transformer.omega(transformer.wsample_fermi()[i], 1);
    }

    dyson_block gb_solver(symm_utils, ns, nts_full, transformer.nw());
    gb_solver.read_input(p.input_file);
    gb_solver.solve(Gb_ts_kij, Sigmafb_ts_kij, 0.0, T_wt, T_tw, w_fermi);

    tensor<dcomplex, 3> G_diff = Gb_ts_kij - sc.G_tau();
    EXPECT_LE(Mcolumn<dcomplex>(G_diff.data(), G_diff.size()).cwiseAbs().maxCoeff(), 1e-10);

    tensor<dcomplex, 3> Sigma_block(nts, ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Sigmab_ts_kij(Sigma_block.data(), nts, ns, symm_utils.flat_ao_size());

    gw_block_solver gw_block(symm_utils, Gb_ts_kij, Sigmab_ts_kij, nts, nts_full, ns, transformer.nw_b(), "flat_");

    gw_block.compute_full_P0_t(df_path);
    for (int q = 0; q < p.nk; ++q) {
      int qslice_size = symm_utils.qaux_slice_sizes()(q);
      int qslice_offset = symm_utils.qaux_slice_offsets()(q);
      for (int t = 0; t < nts; ++t) {
        CMcolumn<dcomplex> P0_compute(sc.solver().P0_q_t_QP().data() + nts_full*qslice_offset + t*qslice_size,
                                      qslice_size);
        CMcolumn<dcomplex> P0_ref(gw_block.P0_t_qQP()(t).data()+qslice_offset, qslice_size);
        auto P0_diff = P0_compute - P0_ref;
        /*std::cout << CMcolumn<dcomplex>(
            gw_block.P0_t_qQP()(t).data(), symm_utils.flat_aux_size()).cwiseAbs().maxCoeff() << std::endl;*/
        //std::cout << P0_diff.cwiseAbs().maxCoeff() << std::endl;
        EXPECT_LE(P0_diff.cwiseAbs().maxCoeff(), 1e-10);
      }
    }
    gw_block.compute_Sigma_from_P_t(df_path, gw_block.P0_t_qQP());
    //gw_block.compute_Sigma_from_P_t(df_path, sc.solver().P0_t_qQP());
    std::cout << "compute reference Sigma done" << std::endl;

    for (int k = 0; k < p.nk; ++k) {
      //std::cout << "k: " << k << std::endl;
      for (int t = 0; t < nts; ++t) {
        for (int s = 0; s < ns; ++s) {
          auto sigma1 = CMcolumn<dcomplex>(sc.Selfenergy()(t, s).data()+symm_utils.kao_slice_offsets()(k),
                                           symm_utils.kao_slice_sizes()(k));
          auto sigma2= CMcolumn<dcomplex>(Sigma_block(t, s).data()+symm_utils.kao_slice_offsets()(k),
                                          symm_utils.kao_slice_sizes()(k));
          auto sigma_diff = sigma1 - sigma2;
          //std::cout << sigma_diff.cwiseAbs().maxCoeff() << std::endl;
          EXPECT_LE(sigma_diff.cwiseAbs().maxCoeff(), 1e-10);
        }
      }
    }
  }
}

TEST(CuGWP, LiHnk2) {

  // solve P0 with cuda solver
  // P0 -> P0_block -> P_block using cpu ref code
  // P0 -> P using cugw_solver

  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  params_t p = symmetry_mbpt::default_parameters();
  p.IR = true;
  p.dfintegral_file = input_block_path + std::string("/df_int");
  p.dfintegral_hf_file = input_block_path + std::string("/df_int");
  p.input_file = input_block_path + std::string("/input_flat.h5");
  p.symmetry_file = input_block_path + std::string("/symmetry_info.h5");
  p.tnc_f_file = ir_path + "/1e3_72.h5";
  p.tnc_b_file = ir_path + "/1e3_72.h5";
  p.ns = 2;
  p.ni = 72;
  p.nk = 8;
  p.beta = 5;
  p.ink = 8;
  p.mu = 0.0;
  p.scf_type = cuGW;
  p.nt_batch = 4;

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();
  //sc.solver().solve();
  sc.solver().solve_P0();

  int NQ = 58;
  tensor<dcomplex, 4> P0_tqQP_block(p.ni+2, p.nk, NQ, NQ);
  tensor<dcomplex, 2> P0_t_qQP(p.ni+2, sc.symm_utils().flat_aux_size());
  for (int q = 0; q < p.nk; ++q) {
    int qslice_size = sc.symm_utils().qaux_slice_sizes()(q);
    int qslice_offset = sc.symm_utils().qaux_slice_offsets()(q);
    for (int t = 0; t < sc.nts(); ++t) {
      CMcolumn<dcomplex> P0_compute(sc.solver().P0_q_t_QP().data() + sc.nts()*qslice_offset + t*qslice_size,
                                    qslice_size);
      Mcolumn<dcomplex> P0_ref(P0_t_qQP(t).data()+qslice_offset, qslice_size);
      P0_ref = P0_compute;
    }
  }
  P_flat_to_block(P0_tqQP_block, P0_t_qQP, sc.symm_utils(), NQ);

  sc.solver().compute_Pq_from_P0q();

  // Block ref
  if (!sc.world_rank()) {
    std::cout << "computing P reference results" << std::endl;
    symmetry_utils_t symm_utils(p);

    int nts_full = p.ni+2;
    int ns = p.ns;
    int nao = 11;

    transformer_t transformer(p.ni, p.beta, p.tnc_f_file, p.tnc_b_file);
    const MatrixX <dcomplex> &T_wt = transformer.Tnt_BF();
    const MatrixX <dcomplex> &T_tw = transformer.Ttn_FB();

    tensor_view<dcomplex, 5> G_tskij(nullptr, nts_full, ns, p.nk, nao, nao);
    tensor_view<dcomplex, 5> Sigma_tskij(nullptr, nts_full, ns, p.nk, nao, nao);

    gw_solver gw(symm_utils, G_tskij, Sigma_tskij, nao, NQ, nts_full, nts_full, ns, transformer.nw_b());
    gw.P0_tqQP().reshape(nts_full, p.nk, NQ, NQ);
    gw.P0_tqQP() = P0_tqQP_block;
    gw.compute_P_from_P0(T_wt, T_tw);

    tensor<dcomplex, 2> P_t_qQP_ref;
    P_block_to_flat(P_t_qQP_ref, gw.P0_tqQP(), symm_utils);

    tensor<dcomplex, 2> P_t_qQP(p.ni+2, sc.symm_utils().flat_aux_size());
    for (int q = 0; q < p.nk; ++q) {
      int qslice_size = sc.symm_utils().qaux_slice_sizes()(q);
      int qslice_offset = sc.symm_utils().qaux_slice_offsets()(q);
      for (int t = 0; t < sc.nts(); ++t) {
        CMcolumn<dcomplex> P_compute(sc.solver().P0_q_t_QP().data() + sc.nts()*qslice_offset + t*qslice_size,
                                      qslice_size);
        Mcolumn<dcomplex> P_ref(P_t_qQP(t).data()+qslice_offset, qslice_size);
        P_ref = P_compute;
      }
    }
    tensor<dcomplex, 1> P_diff(symm_utils.flat_aux_size());
    for (int t = 0; t < nts_full; ++t) {
      P_diff = P_t_qQP(t) - P_t_qQP_ref(t);
      double diff = CMcolumn<dcomplex>(P_diff.data(), symm_utils.flat_aux_size()).cwiseAbs().maxCoeff();
      EXPECT_LE(diff, 1e-9);
    }
  }
}

TEST(CuGWRotP0, LiHnk2) {
  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  params_t p = symmetry_mbpt::default_parameters();
  p.IR = true;
  p.dfintegral_file = input_block_path + std::string("/df_int");
  p.dfintegral_hf_file = input_block_path + std::string("/df_int");
  p.input_file = input_block_path + std::string("/input_flat.h5");
  p.symmetry_file = input_block_path + std::string("/symmetry_info.h5");
  p.tnc_f_file = ir_path + "/1e3_72.h5";
  p.tnc_b_file = ir_path + "/1e3_72.h5";
  p.ns = 2;
  p.ni = 72;
  p.nk = 8;
  p.beta = 5;
  p.ink = 8;
  p.mu = 0.0;
  p.scf_type = cuGW;
  p.nt_batch = 4;

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();
  sc.solver().solve_P0();

  params_t p_rot = p;
  p.ink = 3;
  p_rot.rotate = true;
  p_rot.time_reversal = true;
  p_rot.symmetry_rot_file = input_block_path + std::string("/symmetry_rot.h5");

  sc_loop_block_t sc_rot(MPI_COMM_WORLD, p_rot);
  sc_rot.init_g();
  sc_rot.solver().solve_P0();

  int nt = p.ni + 2;

  if (!sc.world_rank()) {
    // P is in order qtQP
    for (int iq = 0; iq < sc_rot.ink(); ++iq) {
      int q = sc_rot.symm_utils().irre_list()[iq];
      int iq_slice_offset = sc_rot.symm_utils().qaux_slice_offsets_irre()(iq);
      int q_slice_offset = sc_rot.symm_utils().qaux_slice_offsets()(q);
      int iq_slice_size = sc_rot.symm_utils().qaux_slice_sizes_irre()(iq);
      int q_slice_size = sc_rot.symm_utils().qaux_slice_sizes()(q);
      double diff = (CMcolumn<dcomplex>(sc.solver().P0_q_t_QP().data()+nt*q_slice_offset, int(nt/2)*q_slice_size)
                     - CMcolumn<dcomplex>(sc_rot.solver().P0_q_t_QP().data()+nt*iq_slice_offset,
                                          int(nt/2)*iq_slice_size)).norm();
      EXPECT_LE(diff, 1e-10);
    }
  }
}

TEST(CuGWRotSigma, LiHnk2) {
  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  params_t p = symmetry_mbpt::default_parameters();
  p.IR = true;
  p.dfintegral_file = input_block_path + std::string("/df_int");
  p.dfintegral_hf_file = input_block_path + std::string("/df_int");
  p.input_file = input_block_path + std::string("/input_flat.h5");
  p.symmetry_file = input_block_path + std::string("/symmetry_info.h5");
  p.tnc_f_file = ir_path + "/1e3_72.h5";
  p.tnc_b_file = ir_path + "/1e3_72.h5";
  p.ns = 2;
  p.ni = 72;
  p.nk = 8;
  p.beta = 5;
  p.ink = 8;
  p.mu = 0.0;
  p.scf_type = cuGW;
  p.nt_batch = 74;

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();
  sc.solver().solve_sigma();

  params_t p_rot = p;
  p.ink = 3;
  p_rot.rotate = true;
  p_rot.time_reversal = true;
  p_rot.symmetry_rot_file = input_block_path + std::string("/symmetry_rot.h5");

  symmetry_utils_t symm_rot(p_rot);

  sc_loop_block_t sc_rot(MPI_COMM_WORLD, p_rot);
  sc_rot.init_g();
  sc_rot.solver().solve_sigma();

  if (!sc.world_rank()) {
    int ns = p.ns;
    int nt = p.ni+2;
    // compare Pq first
    for (int iq = 0; iq < sc_rot.ink(); ++iq) {
      int q = sc_rot.symm_utils().irre_list()[iq];
      int iq_slice_offset = sc_rot.symm_utils().qaux_slice_offsets_irre()(iq);
      int q_slice_offset = sc_rot.symm_utils().qaux_slice_offsets()(q);
      int iq_slice_size = sc_rot.symm_utils().qaux_slice_sizes_irre()(iq);
      int q_slice_size = sc_rot.symm_utils().qaux_slice_sizes()(q);
      double diff = (CMcolumn<dcomplex>(sc.solver().P0_q_t_QP().data()+nt*q_slice_offset, nt*q_slice_size)
                     - CMcolumn<dcomplex>(sc_rot.solver().P0_q_t_QP().data()+nt*iq_slice_offset,
                                          nt*iq_slice_size)).norm();
      EXPECT_LE(diff, 1e-10);
    }

    // Sigma is in order tskij
    for (int ik = 0; ik < sc_rot.ink(); ++ik) {
      int k = sc_rot.symm_utils().irre_list()[ik];
      int ik_slice_offset = sc_rot.symm_utils().kao_slice_offsets_irre()(ik);
      int k_slice_offset = sc_rot.symm_utils().kao_slice_offsets()(k);
      int ik_slice_size = sc_rot.symm_utils().kao_slice_sizes_irre()(ik);
      int k_slice_size = sc_rot.symm_utils().kao_slice_sizes()(k);
      for (int t = 0; t < 5; ++t) {
        for (int s = 0; s < ns; ++s) {
          double diff = (CMcolumn<dcomplex>(sc.Selfenergy()(t, s).data() + k_slice_offset, k_slice_size)
                         - CMcolumn<dcomplex>(sc_rot.Selfenergy()(t, s).data() + ik_slice_offset, ik_slice_size)).norm();
          EXPECT_LE(diff, 1e-10);
        } // t
      } // s
    } // ik
  }
}
