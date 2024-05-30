#ifndef SYMMETRYMBPT_CUGW_QKPT_BLOCK_H
#define SYMMETRYMBPT_CUGW_QKPT_BLOCK_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cusolverDn.h>

#include "type.h"
#include "cuda_common.h"
#include "cuda_types_map.h"
#include "cublas_routines_prec.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

  template<typename prec>
  class gw_qkpt_block {
    using scalar_t = typename cu_type_map<std::complex<prec> >::cxx_base_type;
    using cxx_complex = typename cu_type_map<std::complex<prec> >::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec> >::cuda_type;
  public:
    gw_qkpt_block(int nao, int naux, int ns, int nt, int nt_batch,
                  const symmetry_utils_t &symm_utils,
                  cuda_complex *g_k_ts_ij, cuda_complex *g_k_mts_ij,
                  cuda_complex *sigma_k_ts_ij, cuda_complex *P_t_qQP, cuda_complex *P0_t_qQP,
                  cuda_complex *kspace_auxrep,
                  cublasHandle_t *handle, int *P0_q_locks, int *sigma_k_locks);

    ~gw_qkpt_block();

    void set_up_qkpt_first(const cxx_complex *V_Qpm_host, int k1, int k2, int q);

    void set_up_qkpt_second(const cxx_complex *V_Qim_host, int k1, int k2, int q);

    void compute_first_tau_contraction();

    void compute_second_tau_contraction();

    bool is_busy();

    void synchronize();

    // TODO: check this
    static std::size_t size(size_t max_kao_size_, size_t max_qaux_size_,
                            size_t max_V_size, size_t nt, size_t nt_batch, size_t ns) {
      return (
                 2 * max_V_size //V_Qpm+V_pmQ
                 + 2 * max_qaux_size_ * nt_batch //local copy of P0
                 + 2 * nt_batch * max_V_size //X1 and X2
                 + ns * nt * max_kao_size_ //sigmak_stij
             ) * sizeof(cuda_complex);
    }

  private:
    // externally handled/allocated Green's functions and self-energies
    cuda_complex *g_k_ts_ij_;
    cuda_complex *g_k_mts_ij_;
    cuda_complex *sigma_k_ts_ij_;
    cuda_complex *P0_q_t_QP_;
    cuda_complex *P_q_t_QP_;
    cuda_complex *kspace_auxrep_;
    int *sigma_k_locks_;
    int *P0_q_locks_;

    // streams
    cudaStream_t stream_;

    // Interaction matrix, density decomposed.
    cuda_complex *V_Qpm_;
    cuda_complex *V_pmQ_;

    // these are two aliases to V matrices to avoid alloc the second time around.
    cuda_complex *V_Qim_;
    cuda_complex *V_nPj_;

    // intermediate vars for strided batched multiplies
    cuda_complex *X2t_Ptm_;
    cuda_complex *X1t_tmQ_;

    // intermediate vars for temp storage of P0
    cuda_complex *Pqk0_tQP_local_; // (nt_batch*naux*naux)
    // intermediate vars for temp storage of P -> need to allocate this buffer for P transform
    cuda_complex *Pqk_tQP_local_; // (nt_batch*naux*naux)

    // intermediate vars for temp storage of sigma and G
    cuda_complex *sigmak_tsij_; // (nt*ns*nao*nao)

    // Pinned host memory for interaction matrix
    cxx_complex *V_Qpm_buffer_;

    int gt_offset_;
    int gmt_offset_;
    int P_offset_;

    // size in ao and aux space
    const int flat_ao_size_;
    const int flat_aux_size_;

    // number of spins
    const int ns_;
    // number of time slices
    const int nt_;
    const int nt_batch_;

    // momentum indices
    int k1_;
    int k2_;
    int q_;
    int k1_slice_offset_;
    int k2_slice_offset_;
    int iq_slice_offset_;
    int k1_slice_size_;
    int k2_slice_size_;
    int iq_slice_size_;
    int q_slice_size_ = 0;
    int V_size_;
    // iBZ related
    int k1_ir_;
    int q_ir_;

    // block sizes and offsets related
    const symmetry_utils_t &symm_utils_;
    const int max_kao_size_;
    const int max_qaux_size_;
    const int max_V_size_;

    //pointer to cublas handle
    cublasHandle_t *handle_;

    void write_P0(int t);
    void write_sigma();

    void get_P(int t);
  };

  template<typename prec>
  gw_qkpt_block<prec> *obtain_idle_qkpt(std::vector<gw_qkpt_block<prec> *> &qkpts) {
    static int pos = 0;
    pos++;
    if (pos >= qkpts.size()) pos = 0;
    while (qkpts[pos]->is_busy()) {
      pos = (pos + 1) % qkpts.size();
    }
    return qkpts[pos];
  }

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_CUGW_QKPT_BLOCK_H
