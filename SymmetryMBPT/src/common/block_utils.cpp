
#include "block_utils.h"

namespace symmetry_mbpt {

void G_block_to_flat(tensor<dcomplex, 3> &G_flat,
                     const tensor<dcomplex, 5> &G_block, const symmetry_utils_t &symm_utils) {

  size_t nts = G_block.shape()[0];
  size_t ns = G_block.shape()[1];
  int nk = G_block.shape()[2];
  int nao = G_block.shape()[3];

  G_flat.reshape({nts, ns, symm_utils.flat_ao_size()});
  G_flat.set_zero();
  for (int t = 0; t < nts; ++t) {
    for (int s = 0; s < ns; ++s) {
      for (int k = 0; k < nk; ++k) {
        int n_block = symm_utils.ao_sizes(k).size();
        // flat indices
        int k_offset = symm_utils.kao_slice_offsets()(k);
        const auto &offsets_k = symm_utils.ao_offsets(k);
        // block indices
        const auto &offsets_bk = symm_utils.ao_block_offsets(k);
        const auto &sizes_k = symm_utils.ao_sizes(k);
        for (int ia = 0; ia < n_block; ++ia) {
          MMatrixX<dcomplex>(G_flat(t, s).data()+k_offset+offsets_k(ia), sizes_k(ia), sizes_k(ia)) =
            CMMatrixX<dcomplex>(G_block(t, s, k).data(), nao, nao).block(offsets_bk(ia), offsets_bk(ia),
                                                                         sizes_k(ia), sizes_k(ia));
        } // ia
      } // k
    } // s
  } // t
}

void aomat_block_to_flat(tensor<dcomplex, 2> &mat_flat,
                         const tensor<dcomplex, 4> &mat_block, const symmetry_utils_t &symm_utils) {

  size_t ns = mat_block.shape()[0];
  int nk = mat_block.shape()[1];
  int nao = mat_block.shape()[2];

  mat_flat.reshape({ns, symm_utils.flat_ao_size()});
  mat_flat.set_zero();
  for (int s = 0; s < ns; ++s) {
    for (int k = 0; k < nk; ++k) {
      int n_block = symm_utils.ao_sizes(k).size();
      // flat indices
      int k_offset = symm_utils.kao_slice_offsets()(k);
      const auto &offsets_k = symm_utils.ao_offsets(k);
      // block indices
      const auto &offsets_bk = symm_utils.ao_block_offsets(k);
      const auto &sizes_k = symm_utils.ao_sizes(k);
      for (int ia = 0; ia < n_block; ++ia) {
        MMatrixX<dcomplex>(mat_flat(s).data()+k_offset+offsets_k(ia), sizes_k(ia), sizes_k(ia)) =
            CMMatrixX<dcomplex>(mat_block(s, k).data(), nao, nao).block(offsets_bk(ia), offsets_bk(ia),
                                                                        sizes_k(ia), sizes_k(ia));
      } // ia
    } // k
  } // s
}

void P_block_to_flat(tensor<dcomplex, 2> &P_flat,
                     const tensor<dcomplex, 4> &P_block, const symmetry_utils_t &symm_utils) {

  size_t nts = P_block.shape()[0];
  int nq = P_block.shape()[1];
  int naux = P_block.shape()[3];

  P_flat.reshape({nts, symm_utils.flat_aux_size()});
  P_flat.set_zero();
  for (int t = 0; t < nts; ++t) {
    for (int q = 0; q < nq; ++q) {
      int n_block = symm_utils.aux_sizes(q).size();
      // flat indices
      int q_offset = symm_utils.qaux_slice_offsets()(q);
      const auto &offsets_q = symm_utils.aux_offsets(q);
      // block indices
      const auto &offsets_bq = symm_utils.aux_block_offsets(q);
      const auto &sizes_q = symm_utils.aux_sizes(q);
      for (int ia = 0; ia < n_block; ++ia) {
        MMatrixX<dcomplex>(P_flat(t).data()+q_offset+offsets_q(ia), sizes_q(ia), sizes_q(ia)) =
          CMMatrixX<dcomplex>(P_block(t, q).data(), naux, naux).block(offsets_bq(ia), offsets_bq(ia),
                                                                      sizes_q(ia), sizes_q(ia));
      } // ia
    } // q
  } // t
}

void G_flat_to_block(tensor<dcomplex, 5> &G_block,
                     const tensor<dcomplex, 3> &G_flat, const symmetry_utils_t &symm_utils, size_t nao) {

  size_t nts = G_flat.shape()[0];
  size_t ns = G_flat.shape()[1];
  size_t nk = symm_utils.nk();

  G_block.reshape({nts, ns, nk, nao, nao});
  G_block.set_zero();
  for (int t = 0; t < nts; ++t) {
    for (int s = 0; s < ns; ++s) {
      for (int k = 0; k < nk; ++k) {
        int n_block = symm_utils.ao_sizes(k).size();
        // flat indices
        int k_offset = symm_utils.kao_slice_offsets()(k);
        const auto &offsets_k = symm_utils.ao_offsets(k);
        // block indices
        const auto &offsets_bk = symm_utils.ao_block_offsets(k);
        const auto &sizes_k = symm_utils.ao_sizes(k);
        for (int ia = 0; ia < n_block; ++ia) {
          MMatrixX<dcomplex>(G_block(t, s, k).data(), nao, nao).block(offsets_bk(ia), offsets_bk(ia),
                                                                      sizes_k(ia), sizes_k(ia)) =
            CMMatrixX<dcomplex>(G_flat(t, s).data()+k_offset+offsets_k(ia), sizes_k(ia), sizes_k(ia));

        } // ia
      } // k
    } // s
  } // t
}

void P_flat_to_block(tensor<dcomplex, 4> &P_block,
                     const tensor<dcomplex, 2> &P_flat, const symmetry_utils_t &symm_utils, size_t naux) {
  size_t nts = P_flat.shape()[0];
  size_t nq = symm_utils.nk();

  P_block.reshape({nts, nq, naux, naux});
  P_block.set_zero();
  for (int t = 0; t < nts; ++t) {
    for (int q = 0; q < nq; ++q) {
      int n_block = symm_utils.aux_sizes(q).size();
      // flat indices
      int q_offset = symm_utils.qaux_slice_offsets()(q);
      const auto &offsets_q = symm_utils.aux_offsets(q);
      // block indices
      const auto &offsets_bq = symm_utils.aux_block_offsets(q);
      const auto &sizes_q = symm_utils.aux_sizes(q);
      for (int ia = 0; ia < n_block; ++ia) {
          MMatrixX<dcomplex>(P_block(t, q).data(), naux, naux).block(offsets_bq(ia), offsets_bq(ia),
                                                                     sizes_q(ia), sizes_q(ia)) =
            CMMatrixX<dcomplex>(P_flat(t).data()+q_offset+offsets_q(ia), sizes_q(ia), sizes_q(ia));
      } // ia
    } // q
  } // t
}

} // namespace symmetry_mbpt