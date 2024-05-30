
#include "mpi_utils.h"
#include "sc_loop_block_t.h"
#include "sc_type.h"

void symmetry_mbpt::sc_loop_block_t::init_g(bool check_convergence) {
  // Intermediate Green's function and density matrix for convergence check
  // only consider first spin
  tensor<dcomplex, 1> dmr_tmp = dmr_(0);
  int gamma_size = symm_utils_.kao_slice_sizes()(0);
  tensor<dcomplex, 2> G_gamma_tmp(nts_, gamma_size);
  for (size_t it = 0; it < nts_; ++it)
    G_gamma_tmp(it) = tensor_view<dcomplex, 1>(G_tau_(it, 0).data(), gamma_size);

  (*sc_type_).compute_G(internode_comm_, node_rank_, world_rank_, world_size_,
                        G_tau_, Selfenergy_, F_k_, H_k_, S_k_, win_G_, mu_);

  dmr_.reshape(ns_, symm_utils_.flat_ao_size());
  dmr_ = G_tau_(nts_ - 1);
  dmr_ *= (ns_ == 2) ? -1.0 : -2.0;
}
