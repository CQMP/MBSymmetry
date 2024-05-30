#ifndef SYMMETRYMBPT_SC_TYPE_H
#define SYMMETRYMBPT_SC_TYPE_H

#include <mpi.h>

#include "common.h"
#include "params_t.h"
#include "symmetry_utils.h"
#include "transformer_t.h"

namespace symmetry_mbpt {

  class sc_type {
  public:

    sc_type(const params_t &params, const transformer_t &ft) : ft_(ft), npoly_(params.ni), nts_(params.ni + 2),
                                                                nw_(params.ni), nk_(params.nk), ink_(params.ink),
                                                                ns_(params.ns){}

    virtual ~sc_type()= default;

    virtual void compute_G(MPI_Comm & intercomm, int node_rank, int world_rank, int world_size,
                           tensor_view<dcomplex, 3> &G_tau, tensor_view<dcomplex, 3> &Sigma_tau,
                           tensor<dcomplex, 2> &F_k, tensor<dcomplex, 2> &H_k, tensor<dcomplex, 2> &S_k,
                           MPI_Win win_G, double mu) const = 0;

  protected:
    // Fourier transform class
    const transformer_t& ft_;
    // number of polynomials
    int npoly_;
    // number of tau points
    int nts_;
    // number of frequency points
    int nw_;
    // number of k-points
    int nk_;
    // number of k-points in the reduced Brillouin zone
    int ink_;
    // number of spins
    int ns_;
  };

  class dyson_sc_block_t : public sc_type {
  public:

    dyson_sc_block_t(const params_t &params, const transformer_t &ft,
                     const symmetry_utils_t &symm_utils) : sc_type(params, ft), symm_utils_(symm_utils) {}

    void compute_G(MPI_Comm & intercomm, int node_rank, int world_rank, int world_size,
                   tensor_view<dcomplex, 3> &G_tau, tensor_view<dcomplex, 3> &Sigma_tau,
                   tensor<dcomplex, 2> &F_k, tensor<dcomplex, 2> &H_k, tensor<dcomplex, 2> &S_k,
                   MPI_Win win_G, double mu) const override;

  private:
    const symmetry_utils_t &symm_utils_;
  };

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_SC_TYPE_H
