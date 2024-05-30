
#include "sc_type.h"

namespace symmetry_mbpt {

  void dyson_sc_block_t::compute_G(MPI_Comm & intercomm, int node_rank, int world_rank, int world_size,
                                   tensor_view<dcomplex, 3> &G_tau, tensor_view<dcomplex, 3> &Sigma_tau,
                                   tensor<dcomplex, 2> &F_k, tensor<dcomplex, 2> &H_k, tensor<dcomplex, 2> &S_k,
                                   MPI_Win win_G, double mu) const {
    // Get G(tau)
    tensor<dcomplex, 2> G_t;
    tensor<dcomplex, 2> G_c;
    tensor<dcomplex, 2> G_w;
    tensor<dcomplex, 2> Sigma_w;
    tensor<dcomplex, 2> Sigma_k;
    Eigen::FullPivLU<MatrixXcd> lusolver;

    MPI_Win_fence(0, win_G);
    if (!node_rank) G_tau.set_zero();
    MPI_Win_fence(0, win_G);

    double coeff_last = 0.0;
    double coeff_first = 0.0;

    for (int isk = world_rank; isk < ns_ * ink_; isk += world_size) {
      int s = isk / ink_;
      int ik = isk % ink_;

      // loop over all blocks for a single k point
      int n_block = symm_utils_.ao_sizes_irre(ik).size();
      int kslice_size = symm_utils_.kao_slice_sizes_irre()(ik);
      int kslice_offset = symm_utils_.kao_slice_offsets_irre()(ik);
      Sigma_k.reshape(nts_, kslice_size);
      Sigma_w.reshape(nw_, kslice_size);
      Sigma_k.set_zero();
      for (int t = 0; t < nts_; ++t) {
        Sigma_k(t).vector() = CMcolumn<dcomplex>(Sigma_tau(t, s).data() + kslice_offset, kslice_size);
      } // t
      ft_.tau_to_omega(Sigma_k, Sigma_w, 1);

      CMcolumn<dcomplex> S(S_k(s).data() + kslice_offset, kslice_size);
      CMcolumn<dcomplex> F(F_k(s).data() + kslice_offset, kslice_size);
      G_w.reshape(nw_, kslice_size);
      G_t.reshape(nts_, kslice_size);
      const auto &offsets_k = symm_utils_.ao_offsets_irre(ik);
      const auto &sizes_k = symm_utils_.ao_sizes_irre(ik);

      for (int ic = 0; ic < nw_; ++ic) {
        std::complex<double> muomega = ft_.omega(ft_.wsample_fermi()[ic], 1) + mu;
        for (int ia = 0; ia < n_block; ++ia) {
          MMatrixX<dcomplex> G(G_w(ic).data()+offsets_k(ia), sizes_k(ia), sizes_k(ia));
          G = (muomega * CMMatrixX<dcomplex>(S.data()+offsets_k(ia), sizes_k(ia), sizes_k(ia))
               - CMMatrixX<dcomplex>(F.data()+offsets_k(ia), sizes_k(ia), sizes_k(ia))
               - MMatrixX<dcomplex>(Sigma_w(ic).data()+offsets_k(ia), sizes_k(ia), sizes_k(ia)));
          G = lusolver.compute(G).inverse().eval();
        } // ia
      } // w

      // Transform back to tau
      ft_.omega_to_tau(G_w, G_t, 1);

      for (int t = 0; t < nts_; ++t) {
        tensor_view<dcomplex, 1>(G_tau(t, s).data() + kslice_offset, kslice_size) = G_t(t);
      } // t

      G_c.reshape(1, kslice_size);
      // Check Chebyshev
      ft_.tau_to_chebyshev_c(G_t, G_c, npoly_ - 1, 1);
      coeff_last = std::max(G_c(0).vector().cwiseAbs().maxCoeff(), coeff_last);
      ft_.tau_to_chebyshev_c(G_t, G_c, 0, 1);
      coeff_first = std::max(G_c(0).vector().cwiseAbs().maxCoeff(), coeff_first);
    } // sk

    MPI_Win_fence(0, win_G);
    if (!node_rank) {
      int MPI_status = MPI_Allreduce(MPI_IN_PLACE, G_tau.data(), G_tau.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, intercomm);
      if (MPI_status != MPI_SUCCESS) {
        std::cout << "Rank " << world_rank << ": MPI_Allreduce fail in init_g() with error " << MPI_status << std::endl;
        throw std::runtime_error("MPI_Allreduce fails in init_g().");
      }
    }
    MPI_Win_fence(0, win_G);

    double leakage = coeff_last / coeff_first;
    if (!world_rank) std::cout << "Leakage of Dyson G: " << leakage << std::endl;
    if (!world_rank and leakage > 1e-8) std::cerr << "Warning: The leakage is larger than 1e-8" << std::endl;
  }

} // namespace symmetry_mbpt