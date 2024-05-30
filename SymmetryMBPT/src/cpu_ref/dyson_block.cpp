
#include "dyson_block.h"

namespace symmetry_mbpt {

void dyson_block::read_input(const std::string &filename) {
  alps::hdf5::archive in_file(filename, "r");
  in_file["HF/Fock-k"] >> F_k_;
  in_file["HF/S-k"] >> S_k_;
  in_file["HF/H-k"] >> H_k_;
  in_file.close();
}

void dyson_block::solve(tensor_view<dcomplex, 3> &G_ts_kij, const tensor_view<dcomplex, 3> &Sigma_ts_kij, double mu,
                        const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
                        const column<dcomplex> &wsample) {
  compute_G(G_ts_kij, Sigma_ts_kij, F_k_, H_k_, S_k_, mu, T_wt, T_tw, wsample);
}

void dyson_block::compute_G(tensor_view<dcomplex, 3> &G_ts_kij, const tensor_view<dcomplex, 3> &Sigma_ts_kij,
                            const tensor<dcomplex, 2> &F_k, const tensor<dcomplex, 2> &H_k,
                            const tensor<dcomplex, 2> &S_k, double mu, const MatrixX<dcomplex> &T_wt,
                            const MatrixX<dcomplex> &T_tw, const column<dcomplex> &wsample) const {
  Eigen::FullPivLU<MatrixXcd> lusolver;

  G_ts_kij.set_zero();
  for (int s = 0; s < ns_; ++s) {
    // loop over all k points
    for (int k = 0; k < symm_utils_.nk(); ++k) {
      // loop over all blocks for a single k point
      int n_block = symm_utils_.ao_sizes(k).size();
      int kslice_size = symm_utils_.kao_slice_sizes()(k);
      int kslice_offset = symm_utils_.kao_slice_offsets()(k);
      tensor<dcomplex, 2> Sigma_k(nts_, kslice_size);
      tensor<dcomplex, 2> Sigma_w(nw_, kslice_size);
      for (int t = 0; t < nts_; ++t) {
        Sigma_k(t).vector() = CMcolumn<dcomplex>(Sigma_ts_kij(t, s).data() + kslice_offset, kslice_size);
      } // t
      MMatrixX<dcomplex>(Sigma_w.data(), nw_, kslice_size)
        = T_wt * MMatrixX<dcomplex>(Sigma_k.data()+kslice_size, nw_, kslice_size);

      CMcolumn<dcomplex> S(S_k(s).data() + kslice_offset, kslice_size);
      CMcolumn<dcomplex> F(F_k(s).data() + kslice_offset, kslice_size);

      tensor<dcomplex, 2> G_w(nw_, kslice_size);
      tensor<dcomplex, 2> G_t(nts_, kslice_size);
      const auto &offsets_k = symm_utils_.ao_offsets(k);
      const auto &sizes_k = symm_utils_.ao_sizes(k);

      for (int w = 0; w < nw_; ++w) {
        std::complex<double> muomega = wsample(w) + mu;
        for (int ia = 0; ia < n_block; ++ia) {
          MMatrixX<dcomplex> G(G_w(w).data()+offsets_k(ia), sizes_k(ia), sizes_k(ia));
          G = (muomega * CMMatrixX<dcomplex>(S.data()+offsets_k(ia), sizes_k(ia), sizes_k(ia))
            - CMMatrixX<dcomplex>(F.data()+offsets_k(ia), sizes_k(ia), sizes_k(ia))
            - MMatrixX<dcomplex>(Sigma_w(w).data()+offsets_k(ia), sizes_k(ia), sizes_k(ia)));
          G = lusolver.compute(G).inverse().eval();
        } // ia
      } // w
      MMatrixX<dcomplex>(G_t.data(), nts_, kslice_size) = T_tw * MMatrixX<dcomplex>(G_w.data(), nw_, kslice_size);
      for (int t = 0; t < nts_; ++t) {
        tensor_view<dcomplex, 1>(G_ts_kij(t, s).data() + kslice_offset, kslice_size) = G_t(t);
      } // t
    } // k
  } // s
}

} // namespace symmetry_mbpt
