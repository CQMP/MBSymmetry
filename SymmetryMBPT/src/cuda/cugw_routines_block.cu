
#include "cugw_routines_block.h"

namespace symmetry_mbpt {

  template<typename prec>
  cugw_routine_block<prec>::cugw_routine_block(int nk, int ink, int ns,
                                               int nt_batch, int nt,
                                               const tensor<long, 2> &qk_pairs, const symmetry_utils_t &symm_utils,
                                               std::complex<double> *Vk1k2_Qij,
                                               int nqkpt,
                                               int devices_rank, int devices_size,
                                               int myid, int intranode_rank, int devCount_per_node):
      nk_(nk), ink_(ink), ns_(ns),
      nt_batch_(nt_batch), nt_(nt),
      n_qkpair_(qk_pairs.shape()[0]),
      qk_pairs_(qk_pairs),
      symm_utils_(symm_utils),
      flat_ao_size_(symm_utils.flat_ao_size()),
      flat_aux_size_(symm_utils.flat_aux_size()),
      flat_orep_size_(symm_utils.flat_orep_size()),
      flat_auxrep_size_(symm_utils.flat_auxrep_size()),
      flat_ao_size_full_(symm_utils.flat_ao_size_full()),
      Vk1k2_Qij_(Vk1k2_Qij),
      nqkpt_(nqkpt),
      devices_rank_(devices_rank), devices_size_(devices_size) {
    if (cudaSetDevice(intranode_rank % devCount_per_node) != cudaSuccess)
      throw std::runtime_error("Error in cudaSetDevice2");
    if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(myid) + ": error initializing cublas");

    allocate_memory();
    set_locks();

    // Each process gets one cuda runner for qk pair
    qkpts_.resize(nqkpt_);
    for(int i = 0; i < nqkpt_; ++i){
      qkpts_[i]=new gw_qkpt_block<prec>(0, 0, ns_, nt_, nt_batch_, symm_utils_,
                                        g_k_ts_ij_device_, g_k_mts_ij_device_, sigma_k_ts_ij_device_,
                                        P_q_t_QP_device_, P0_q_t_QP_device_,
                                        kspace_auxrep_device_,
                                        &handle_, P0_q_locks_, sigma_k_locks_);
    }
  }

  template<typename prec>
  cugw_routine_block<prec>::~cugw_routine_block() {
    for(int i = 0; i < qkpts_.size(); ++i){
      delete qkpts_[i];
    }
    cudaFree(g_k_ts_ij_device_);
    cudaFree(g_k_mts_ij_device_);
    cudaFree(P_q_t_QP_device_);
    cudaFree(kspace_auxrep_device_);
    cudaFree(kspace_orep_device_);
    cudaFree(sigma_k_locks_);
    cudaFree(P0_q_locks_);
    if(cublasDestroy(handle_)!=CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublas error destroying handle");
  }

  template<typename prec>
  void cugw_routine_block<prec>::allocate_memory() {
    if(cudaMalloc(&g_k_ts_ij_device_, nt_*ns_*flat_ao_size_full_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("G could not be allocated");
    if(cudaMalloc(&g_k_mts_ij_device_, nt_*ns_*flat_ao_size_full_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("-G could not be allocated");
    if(cudaMalloc(&P_q_t_QP_device_, nt_*flat_aux_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("P could not be allocated");
    if(cudaMalloc(&kspace_auxrep_device_, flat_auxrep_size_*sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("kspace_auxrep could not be allocated");
    if(cudaMalloc(&kspace_orep_device_, flat_orep_size_*sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("kspace_orep could not be allocated");

    // set memory alias
    sigma_k_ts_ij_device_ = g_k_mts_ij_device_;
    P0_q_t_QP_device_ = P_q_t_QP_device_;

    cudaMemset(P0_q_t_QP_device_, 0, (nt_)*flat_aux_size_*sizeof(cuda_complex));
  }

  template<typename prec>
  void cugw_routine_block<prec>::set_locks() {
    // locks so that different threads don't write the results over each other
    cudaMalloc(&sigma_k_locks_, ink_ * sizeof(int));
    cudaMemset(sigma_k_locks_, 0, ink_ * sizeof(int));

    // locks so that different threads don't write the results over each other
    cudaMalloc(&P0_q_locks_, ink_ * sizeof(int));
    cudaMemset(P0_q_locks_, 0, ink_ * sizeof(int));
  }

  template<typename prec>
  void cugw_routine_block<prec>::synchronize_streams() {
    for(int i = 0; i < qkpts_.size(); ++i){
      qkpts_[i]->synchronize();
    }
  }

  template<typename prec>
  void cugw_routine_block<prec>::copy_G_host_to_device(const std::complex<double> *G_ts_kij_host) {

    std::size_t size = ns_ * nt_ * flat_ao_size_; // TODO: allocate smaller buffer if out of memory
    std::vector<cxx_complex> G_k_ts_ij(size);
    // change order of indices
    for (int ik = 0; ik < ink_; ++ik) {
      int slice_size = symm_utils_.kao_slice_sizes_irre()(ik);
      int offset = symm_utils_.kao_slice_offsets_irre()(ik);
      for (int t = 0; t < nt_; ++t) {
        for (int s = 0; s < ns_; ++s) {
          int ts = t * ns_ + s;
          int tskij_shift = ts * flat_ao_size_ + offset;
          int ktsij_shift = ns_ * nt_ * offset + ts * slice_size;
          std::transform(G_ts_kij_host+tskij_shift, G_ts_kij_host+tskij_shift+slice_size, G_k_ts_ij.data()+ktsij_shift,
                         [&](const std::complex<double> &in) { return static_cast<cxx_complex>(in); });
        }
      }
    }
    // use g(-t) as temp buffer
    if (cudaMemcpy(g_k_mts_ij_device_, G_k_ts_ij.data(), size * sizeof(cxx_complex),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("failure to copy in G");

    // -- rotate G to get value in full BZ
    // allocate one extra buffer
    cuda_complex *G_tmp_device;
    // TODO: only need to allocate largest block size in orep
    if(cudaMalloc(&G_tmp_device, nt_*ns_*symm_utils_.max_kao_size()*sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("G temp could not be allocated");
    int two = 2;
    scalar_t alpha = -1;
    cuda_complex one = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    for (int k = 0; k < nk_; ++k) {
      int ck = symm_utils_.conj_index()[k];
      int ick = symm_utils_.irre_index()[ck];
      int kslice_size = symm_utils_.kao_slice_sizes()(k);
      int ikslice_size = symm_utils_.kao_slice_sizes_irre()(ick);
      int kslice_offset = symm_utils_.kao_slice_offsets()(k)*ns_*nt_;
      int ikslice_offset = symm_utils_.kao_slice_offsets_irre()(ick)*ns_*nt_;

      if (symm_utils_.index()[ck] == ck) {
        if (cudaMemcpy(g_k_ts_ij_device_+kslice_offset, g_k_mts_ij_device_+ikslice_offset,
                       kslice_size*ns_*nt_ * sizeof(cuda_complex),
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
          throw std::runtime_error("failure to copy G from iBZ to fBZ");
      } else {
        const auto &offsets_k = symm_utils_.ao_offsets(ck);
        const auto &sizes_k = symm_utils_.ao_sizes(ck);
        const auto &offsets_ik = symm_utils_.ao_offsets_irre(ick);
        const auto &sizes_ik = symm_utils_.ao_sizes_irre(ick);
        int orepslice_offset = symm_utils_.korep_slice_offsets()(ck);
        const auto &irreps = symm_utils_.orep_irreps()[ck];
        const auto &offsets = symm_utils_.orep_offsets()[ck];
        // orep * dmm * orep.adjoint();
        for (int ia = 0; ia < irreps.shape()[0]; ++ia) {
          int ia1 = irreps(ia, 0);
          int ia2 = irreps(ia, 1);
          int k_size = sizes_k(ia1);
          int ik_size = sizes_ik(ia2);
          int k_offset = offsets_k(ia1);
          int ik_offset = offsets_ik(ia2);
          int o_offset = offsets(ia);
          if (GEMM_STRIDED_BATCHED(handle_, CUBLAS_OP_N, CUBLAS_OP_N, ik_size, k_size, ik_size, &one,
                                   g_k_mts_ij_device_ + ikslice_offset + ik_offset, ik_size, ikslice_size,
                                   kspace_orep_device_+orepslice_offset+o_offset, ik_size, 0,
                                   &zero, G_tmp_device, ik_size, k_size*ik_size,
                                   nt_*ns_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on copy_G_host_to_device().");
          }
          if (GEMM_STRIDED_BATCHED(handle_, CUBLAS_OP_C, CUBLAS_OP_N, k_size, k_size, ik_size, &one,
                                   kspace_orep_device_+orepslice_offset+o_offset, ik_size, 0,
                                   G_tmp_device, ik_size, k_size*ik_size,
                                   &zero, g_k_ts_ij_device_ + kslice_offset + k_offset, k_size, kslice_size,
                                   nt_*ns_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on copy_G_host_to_device().");
          }
        } // ia
      } // rotate
      if (k != ck) {
        if (RSCAL(handle_, kslice_size*ns_*nt_, &alpha,
                  (scalar_t *) g_k_ts_ij_device_ + kslice_offset + 1, two) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("RSCAL fails on copy_G_host_to_device().");
        }
      } // conjugate
    } // k

    // get G(-t)
    for (int k = 0; k < nk_; ++k) {
      int kslice_size = symm_utils_.kao_slice_sizes()(k);
      int kslice_offset = symm_utils_.kao_slice_offsets()(k)*ns_*nt_;
      for (std::size_t t = 0; t < nt_; ++t) {
        std::size_t shift_t = t * ns_ * kslice_size;
        std::size_t shift_mt = (nt_ - 1 - t) * ns_ * kslice_size;
        if (cudaMemcpy(g_k_mts_ij_device_ + kslice_offset + shift_mt, g_k_ts_ij_device_ + kslice_offset + shift_t,
                       ns_ * kslice_size * sizeof(cuda_complex), cudaMemcpyDeviceToDevice) != cudaSuccess)
          throw std::runtime_error("failure to copy G to -G");
      }
    }
    cudaFree(G_tmp_device);
  }

  template<typename prec>
  void cugw_routine_block<prec>::copy_P_host_to_device(const std::complex<double> *P_q_t_QP_host) {
    std::size_t size = nt_ * flat_aux_size_;
    std::vector<cxx_complex> P_tmp(size);  // TODO: allocate a smaller buffer if run out of memory
    std::transform(P_q_t_QP_host, P_q_t_QP_host + size, P_tmp.data(),
                   [&](const std::complex<double> &in) { return static_cast<cxx_complex>(in); });
    if (cudaMemcpy(P_q_t_QP_device_, P_tmp.data(), size * sizeof(cxx_complex),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("failure to copy in P");
  }

  template<typename prec>
  void cugw_routine_block<prec>::copy_AuxRep_host_to_device(const std::complex<double> *AuxRep_kQP_host) {
    std::size_t size = flat_auxrep_size_;
    std::vector<cxx_complex> AuxRep_tmp(size);
    std::transform(AuxRep_kQP_host, AuxRep_kQP_host + size, AuxRep_tmp.data(),
                   [&](const std::complex<double> &in) { return static_cast<cxx_complex>(in); });
    if (cudaMemcpy(kspace_auxrep_device_, AuxRep_tmp.data(), size * sizeof(cxx_complex),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("failure to copy in AuxRep");
  }

  template<typename prec>
  void cugw_routine_block<prec>::copy_ORep_host_to_device(const std::complex<double> *ORep_kij_host) {
    std::size_t size = flat_orep_size_;
    std::vector<cxx_complex> ORep_tmp(size);
    std::transform(ORep_kij_host, ORep_kij_host + size, ORep_tmp.data(),
                   [&](const std::complex<double> &in) { return static_cast<cxx_complex>(in); });
    if (cudaMemcpy(kspace_orep_device_, ORep_tmp.data(), size * sizeof(cxx_complex),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("failure to copy in ORep");
  }

  template<typename prec>
  void cugw_routine_block<prec>::copy_P0_device_to_host(std::complex<double> *P0_q_t_QP_host) const {
    std::size_t size = nt_ * flat_aux_size_;
    std::vector<cxx_complex> P0_tmp(size);  // TODO: allocate a smaller buffer if run out of memory
    if (cudaMemcpy(P0_tmp.data(), P0_q_t_QP_device_, size * sizeof(cxx_complex),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("failure to copy P0 to host");
    std::transform(P0_q_t_QP_host, P0_q_t_QP_host + size, P0_tmp.data(), P0_q_t_QP_host,
                   [&](const std::complex<double> &A, const cxx_complex &B) {
                     return A + static_cast<std::complex<double> >(B);
                   });
  }

  template<typename prec>
  void cugw_routine_block<prec>::copy_Sigma_device_to_host(std::complex<double> *Sigma_ts_kij_host) const {
    std::size_t size = nt_ * ns_ * flat_ao_size_;
    std::vector<cxx_complex> sigma_tmp(size); // TODO: allocate a smaller buffer if run out of memory
    if (cudaMemcpy(sigma_tmp.data(), sigma_k_ts_ij_device_, size * sizeof(cxx_complex),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("failure to copy Sigma to host");

    for (int ik = 0; ik < ink_; ++ik) {
      int slice_size = symm_utils_.kao_slice_sizes_irre()(ik);
      int offset = symm_utils_.kao_slice_offsets_irre()(ik);
      for (int t = 0; t < nt_; ++t) {
        for (int s = 0; s < ns_; ++s) {
          int ts = t * ns_ + s;
          int tskij_shift = ts * flat_ao_size_ + offset;
          int ktsij_shift = ns_ * nt_ * offset + ts * slice_size;
          std::transform(Sigma_ts_kij_host+tskij_shift, Sigma_ts_kij_host+tskij_shift+slice_size,
                         sigma_tmp.data()+ktsij_shift,
                         Sigma_ts_kij_host+tskij_shift,
                         [&](const std::complex<double> &A, const cxx_complex &B) {
                           return A + static_cast<std::complex<double> >(B);
                         });
        }
      }
    }
  }

  template<typename prec>
  void cugw_routine_block<prec>::solve_P0(mom_cons_callback &momentum_conservation,
                                          integral_reader_callback<prec> &reader) {
    for (std::size_t i = devices_rank_; i < n_qkpair_; i += devices_size_) {
      size_t q_iBZ = qk_pairs_(i, 0);  // index in iBZ
      size_t q = qk_pairs_(i, 1); // index in full BZ
      size_t k = qk_pairs_(i, 2); // index in full BZ
      std::array<size_t, 4> k_vector = momentum_conservation({{k, 0, q}});
      size_t k1 = k_vector[3]; // k1 = k + q
      // TODO: later on we should find k and k1 in iBZ. q is already in iBZ.
      // prepare interaction tensors before solving
      reader(k, k1, V_Qpm_, Vk1k2_Qij_);

      gw_qkpt_block<prec> *qkpt = obtain_idle_qkpt(qkpts_);
      qkpt->set_up_qkpt_first(V_Qpm_.data(), k, k1, q_iBZ);
      qkpt->compute_first_tau_contraction();
    }
    synchronize_streams();
  }

  template<typename prec>
  void cugw_routine_block<prec>::solve_Sigma(mom_cons_callback &momentum_conservation,
                                             integral_reader_callback<prec> &reader) {

    cudaMemset(sigma_k_ts_ij_device_, 0, nt_*ns_*flat_ao_size_*sizeof(cuda_complex));

    for (std::size_t i = devices_rank_; i < n_qkpair_; i += devices_size_) {
      size_t k_iBZ = qk_pairs_(i, 0); // index in iBZ
      size_t k = qk_pairs_(i, 1); // index in full BZ
      size_t q = qk_pairs_(i, 2); // index in full BZ
      //size_t q_iBZ = irre_index_[q]; // index in iBZ for corresponding iBZ(q)
      std::array<size_t, 4> k_vector = momentum_conservation({{k, q, 0}});
      size_t k1 = k_vector[3]; // k1=k-q
      // TODO: later on we should find k and k1 in iBZ. q is already in iBZ.
      // prepare interaction tensors before solving
      reader(k, k1, V_Qim_, Vk1k2_Qij_); // read in order V_Pnj

      gw_qkpt_block<prec> *qkpt = obtain_idle_qkpt(qkpts_);
      qkpt->set_up_qkpt_second(V_Qim_.data(), k, k1, q);
      qkpt->compute_second_tau_contraction();
    }
    synchronize_streams();
  }

  template class cugw_routine_block<float>;
  template class cugw_routine_block<double>;

} // namespace symmetry_mbpt