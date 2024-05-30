#ifndef SYMMETRYMBPT_CUGW_ROUTINES_BLOCK_H
#define SYMMETRYMBPT_CUGW_ROUTINES_BLOCK_H

#include <cstring>

#include "type.h"

#include "cublas_routines_prec.h"
#include "cuda_common.h"
#include "cugw_qkpt_block.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

  using mom_cons_callback = std::function<std::array<size_t, 4>(const std::array<size_t, 3> &)>;

  template<typename prec>
  using integral_reader_callback = std::function<void(int k, int k1,
                                                      tensor<std::complex<prec>, 1> &,
                                                      std::complex<double> *)>;

  template <typename prec>
  class cugw_routine_block {
    using scalar_t = typename cu_type_map<std::complex<prec> >::cxx_base_type;
    using cxx_complex = typename cu_type_map<std::complex<prec> >::cxx_type;
    using cuda_complex = typename cu_type_map<std::complex<prec> >::cuda_type;
  public:
    cugw_routine_block(int nk, int ink, int ns, int nt_batch, int nt,
                       const tensor<long, 2> &qk_pairs, const symmetry_utils_t &symm_utils,
                       std::complex<double> *Vk1k2_Qij,
                       int nqkpt,
                       int devices_rank, int devices_size,
                       int myid, int intranode_rank, int devCount_per_node);

    ~cugw_routine_block();

    void copy_G_host_to_device(const std::complex<double> *G_tskij_host);

    void copy_P_host_to_device(const std::complex<double> *P_q_t_QP_host);

    void copy_AuxRep_host_to_device(const std::complex<double> *AuxRep_kQP_host);

    void copy_ORep_host_to_device(const std::complex<double> *ORep_kij_host);

    void copy_P0_device_to_host(std::complex<double> *P0_q_t_QP_host) const;

    void copy_Sigma_device_to_host(std::complex<double> *Sigma_tskij_host) const;

    void solve_P0(mom_cons_callback &momentum_conservation, integral_reader_callback<prec> &reader);

    void solve_Sigma(mom_cons_callback &momentum_conservation, integral_reader_callback<prec> &reader);

  private:
    const int nk_;
    const int ink_;
    const int ns_;
    const int nt_batch_;
    const int nt_;
    const int n_qkpair_;
    const tensor<long, 2> &qk_pairs_;
    const symmetry_utils_t &symm_utils_;
    const int flat_ao_size_;
    const int flat_aux_size_;
    const int flat_orep_size_;
    const int flat_auxrep_size_;
    const int flat_ao_size_full_;

    // qkpts on a GPU device
    std::vector<gw_qkpt_block<prec>*> qkpts_;
    const int nqkpt_;

    const int devices_rank_;
    const int devices_size_;

    // alias to pointer of interaction tensor
    std::complex<double> *Vk1k2_Qij_;

    // interaction tensors needed for computing P0
    tensor<std::complex<prec>, 1> V_Qpm_;
    tensor<std::complex<prec>, 1> V_Qpm_buffer_; // buffer for transpose
    tensor<std::complex<prec>, 1> V_pmQ_;
    // interaction tensors needed for computing sigma, reuse memory
    tensor<std::complex<prec>, 1> &V_Qim_ = V_Qpm_;
    tensor<std::complex<prec>, 1> &V_njP_buffer_ = V_Qpm_buffer_;
    tensor<std::complex<prec>, 1> &V_nPj_ = V_pmQ_;

    // memories need to be allocated
    // TODO: temporarily we allocate G for all k points
    cuda_complex *g_k_ts_ij_device_; // (nt*ns*flat_ao_size)
    cuda_complex *g_k_mts_ij_device_; // (nt*ns*flat_ao_size)
    cuda_complex *P_q_t_QP_device_; // (nt*flat_aux_size)

    // transform matrix in auxiliary orbital space
    cuda_complex *kspace_auxrep_device_; // (flat_auxrep_size)
    // transform matrix in orbital space
    cuda_complex *kspace_orep_device_; // (flat_orep_size)

    // this should be an alias to g_ksmtij_device_, even though we only need sigma in iBZ
    cuda_complex *sigma_k_ts_ij_device_;
    // this should be an alias to P_q_t_QP_device_
    cuda_complex *P0_q_t_QP_device_;

    // write locks for P0 and Sigma.
    // In general, they could use the same memories, but they should be small enough to be allocated twice
    int *sigma_k_locks_;
    int *P0_q_locks_;

    cublasHandle_t handle_;

    void allocate_memory();
    void set_locks();

    void synchronize_streams();
  };

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_CUGW_ROUTINES_BLOCK_H
