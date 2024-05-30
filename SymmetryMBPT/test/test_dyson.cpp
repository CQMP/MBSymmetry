
// compute block dyson and compare with full dyson

#include "gtest/gtest.h"

#include "type.h"
#include "symmetry_utils.h"
#include "transformer_t.h"

#include "dyson.h"
#include "dyson_block.h"
#include "block_utils.h"

#include "sc_loop_block_t.h"


TEST(Dyson, BlockDyson) {

  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  int ns = 2;
  int nts_full = 74;
  int nk = 8;
  int nao = 11;
  int ni = 72;
  int beta = 5;

  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  transformer_t transformer(ni, beta, ir_path+"/1e3_72.h5", ir_path+"/1e3_72.h5");
  const MatrixX<dcomplex> &T_wt = transformer.Tnt();
  const MatrixX<dcomplex> &T_tw = transformer.Ttn();
  column<dcomplex> w_fermi(ni);
  for (int i = 0; i < ni; ++i) {
    w_fermi(i) = transformer.omega(transformer.wsample_fermi()[i], 1);
  }

  /*
   * Reference
   */
  tensor<dcomplex, 5> G_full(nts_full, ns, nk, nao, nao);
  tensor_view<dcomplex, 5> G_tskij(G_full.data(), nts_full, ns, nk, nao, nao);

  tensor<dcomplex, 5> Sigma_full(nts_full, ns, nk, nao, nao);
  tensor_view<dcomplex, 5> Sigma_full_tskij(Sigma_full.data(), nts_full, ns, nk, nao, nao);
  Sigma_full_tskij.set_zero();

  dyson g_solver(ns, nts_full, transformer.nw(), nk, nao);
  g_solver.read_input(input_block_path + std::string("/input_blk.h5"));
  g_solver.solve(G_tskij, Sigma_full_tskij, 0.0, T_wt, T_tw, w_fermi);

  /*
   * Block
   */
  std::string df_path = input_path + std::string("/df_int");
  symmetry_utils_t symm_utils(df_path, nk, input_block_path+"/symmetry_info.h5");

  tensor<dcomplex, 3> G_block(nts_full, ns, symm_utils.flat_ao_size());
  tensor_view<dcomplex, 3> Gb_ts_kij(G_block.data(), nts_full, ns, symm_utils.flat_ao_size());

  tensor<dcomplex, 3> Sigma_full_block(nts_full, ns, symm_utils.flat_ao_size());
  tensor_view<dcomplex, 3> Sigmafb_ts_kij(Sigma_full_block.data(), nts_full, ns, symm_utils.flat_ao_size());
  Sigmafb_ts_kij.set_zero();

  dyson_block gb_solver(symm_utils, ns, nts_full, transformer.nw());
  gb_solver.read_input(input_block_path + std::string("/input_flat.h5"));
  gb_solver.solve(Gb_ts_kij, Sigmafb_ts_kij, 0.0, T_wt, T_tw, w_fermi);

  /*
   * Compare
   */
  tensor<dcomplex, 3> G_flat;
  G_block_to_flat(G_flat, G_tskij, symm_utils);
  tensor<dcomplex, 5> G_block_back;
  G_flat_to_block(G_block_back, G_flat, symm_utils, nao);
  tensor<dcomplex, 5> transform_diff = G_block_back - G_tskij;
  EXPECT_LE(Mcolumn<dcomplex>(transform_diff.data(), transform_diff.size()).cwiseAbs().maxCoeff(), 1e-12);
  tensor<dcomplex, 3> G_diff = G_flat - G_block;
  EXPECT_LE(Mcolumn<dcomplex>(G_diff.data(), G_diff.size()).cwiseAbs().maxCoeff(), 1e-10);
}

TEST(Dyson, SCDyson) {

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
  p.scf_type = cuHF;

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();

  /*
   * Block ref
   */
  if (!sc.world_rank()) {
    symmetry_utils_t symm_utils(p);

    tensor<dcomplex, 3> G_block(p.ni+2, p.ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Gb_ts_kij(G_block.data(), p.ni+2, p.ns, symm_utils.flat_ao_size());

    tensor<dcomplex, 3> Sigma_full_block(p.ni+2, p.ns, symm_utils.flat_ao_size());
    tensor_view<dcomplex, 3> Sigmafb_ts_kij(Sigma_full_block.data(), p.ni+2, p.ns, symm_utils.flat_ao_size());
    Sigmafb_ts_kij.set_zero();

    transformer_t transformer(p.ni, p.beta, p.tnc_f_file, p.tnc_b_file);
    const MatrixX<dcomplex> &T_wt = transformer.Tnt();
    const MatrixX<dcomplex> &T_tw = transformer.Ttn();
    column<dcomplex> w_fermi(p.ni);
    for (int i = 0; i < p.ni; ++i) {
      w_fermi(i) = transformer.omega(transformer.wsample_fermi()[i], 1);
    }
    dyson_block gb_solver(symm_utils, p.ns, p.ni+2, transformer.nw());
    gb_solver.read_input(p.input_file);
    gb_solver.solve(Gb_ts_kij, Sigmafb_ts_kij, 0.0, T_wt, T_tw, w_fermi);

    tensor<dcomplex, 3> G_diff = Gb_ts_kij - sc.G_tau();
    EXPECT_LE(Mcolumn<dcomplex>(G_diff.data(), G_diff.size()).cwiseAbs().maxCoeff(), 1e-10);
  }
}

TEST(Dyson, IrreDyson) {
  using namespace symmetry_mbpt;

  // LiH with 2 atoms in a unit cell
  std::string input_path = std::string(INTEGRALS_PATH) + "/LiH/integral/nk2";
  std::string input_block_path = input_path + std::string("/block");
  std::string ir_path = std::string(INTEGRALS_PATH) + "/ir_grid";

  // no rotation
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
  p.scf_type = cuHF;
  p.symmetry_rot_file = "";

  sc_loop_block_t sc(MPI_COMM_WORLD, p);
  sc.init_g();

  // rotation
  params_t p_rot = p;
  p.ink = 3;
  p_rot.rotate = true;
  p_rot.time_reversal = true;
  p_rot.symmetry_rot_file = input_block_path + std::string("/symmetry_rot.h5");

  sc_loop_block_t sc_rot(MPI_COMM_WORLD, p_rot);
  sc_rot.init_g();

  if (!sc.world_rank()) {

    // check Fock first to make sure inputs are correct
    for (int s = 0; s < sc.ns(); ++s) {
      for (int ik = 0; ik < sc_rot.ink(); ++ik) {
        int k = sc_rot.symm_utils().irre_list()[ik];
        int ik_slice_offset = sc_rot.symm_utils().kao_slice_offsets_irre()(ik);
        int k_slice_offset = sc_rot.symm_utils().kao_slice_offsets()(k);
        int ik_slice_size = sc_rot.symm_utils().kao_slice_sizes_irre()(ik);
        int k_slice_size = sc_rot.symm_utils().kao_slice_sizes()(k);
        double diff = (CMcolumn<dcomplex>(sc.F_k()(s).data()+k_slice_offset, k_slice_size)
                       - CMcolumn<dcomplex>(sc_rot.F_k()(s).data()+ik_slice_offset, ik_slice_size)).norm();
        EXPECT_LE(diff, 1e-10);
      }
    }

    for (int s = 0; s < sc.ns(); ++s) {
      for (int t = 0; t < p.ni+2; ++t) {
        for (int ik = 0; ik < sc_rot.ink(); ++ik) {
          int k = sc_rot.symm_utils().irre_list()[ik];
          int ik_slice_offset = sc_rot.symm_utils().kao_slice_offsets_irre()(ik);
          int k_slice_offset = sc_rot.symm_utils().kao_slice_offsets()(k);
          int ik_slice_size = sc_rot.symm_utils().kao_slice_sizes_irre()(ik);
          int k_slice_size = sc_rot.symm_utils().kao_slice_sizes()(k);
          double diff = (CMcolumn<dcomplex>(sc.G_tau()(s, t).data()+k_slice_offset, k_slice_size)
              - CMcolumn<dcomplex>(sc_rot.G_tau()(s, t).data()+ik_slice_offset, ik_slice_size)).norm();
          EXPECT_LE(diff, 1e-10);
        }
      }
    }
  } // world rank
}
