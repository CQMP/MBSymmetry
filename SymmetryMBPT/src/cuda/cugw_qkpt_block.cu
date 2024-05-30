
#include "cugw_qkpt_block.h"

namespace symmetry_mbpt {

  template<typename prec>
  gw_qkpt_block<prec>::gw_qkpt_block(int nao, int naux, int ns, int nt, int nt_batch,
                                     const symmetry_utils_t &symm_utils,
                                     cuda_complex *g_k_ts_ij, cuda_complex *g_k_mts_ij,
                                     cuda_complex *sigma_k_ts_ij, cuda_complex *P_t_qQP, cuda_complex *P0_t_qQP,
                                     cuda_complex *kspace_auxrep,
                                     cublasHandle_t *handle, int *P0_q_locks, int *sigma_k_locks):
      g_k_ts_ij_(g_k_ts_ij),
      g_k_mts_ij_(g_k_mts_ij),
      sigma_k_ts_ij_(sigma_k_ts_ij),
      P0_q_t_QP_(P0_t_qQP),
      P_q_t_QP_(P_t_qQP),
      kspace_auxrep_(kspace_auxrep),
      P0_q_locks_(P0_q_locks),
      sigma_k_locks_(sigma_k_locks),
      flat_ao_size_(symm_utils.flat_ao_size()),
      flat_aux_size_(symm_utils.flat_aux_size()),
      max_kao_size_(symm_utils.max_kao_size()),
      max_qaux_size_(symm_utils.max_qaux_size()),
      max_V_size_(symm_utils.max_kpair_size()),
      ns_(ns),
      nt_(nt),
      nt_batch_(nt_batch),
      symm_utils_(symm_utils),
      handle_(handle) {
    if (cudaStreamCreate(&stream_) != cudaSuccess) throw std::runtime_error("main stream creation failed");

    //interaction matrix and its transpose
    if (cudaMalloc(&V_Qpm_, max_V_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on device");
    if (cudaMalloc(&V_pmQ_, max_V_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on device");

    //intermediate vars for strided batched multiplies  TODO: check this, only need to be max block size
    if (cudaMalloc(&X1t_tmQ_, nt_batch_ * max_V_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating X1 on device");
    if (cudaMalloc(&X2t_Ptm_, nt_batch_ * max_V_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating X2 on device");

    //buffer for self-energy and Green's function
    if (cudaMalloc(&sigmak_tsij_, nt_ * ns_ * max_kao_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating sigma on device");

    if (cudaMalloc(&Pqk0_tQP_local_, nt_batch_ * max_qaux_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Pq0");
    if (cudaMalloc(&Pqk_tQP_local_, nt_batch_ * max_qaux_size_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Pq");

    if (cudaMallocHost(&V_Qpm_buffer_, max_V_size_ * sizeof(cxx_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on host");

    //set memory alias
    V_Qim_ = V_Qpm_;
    V_nPj_ = V_pmQ_;
  }

  template<typename prec>
  gw_qkpt_block<prec>::~gw_qkpt_block() {
    cudaStreamDestroy(stream_);

    cudaFree(V_Qpm_);
    cudaFree(V_pmQ_);
    cudaFree(X1t_tmQ_);
    cudaFree(X2t_Ptm_);
    cudaFree(Pqk0_tQP_local_);
    cudaFree(Pqk_tQP_local_);
    cudaFree(sigmak_tsij_);

    cudaFreeHost(V_Qpm_buffer_);
  }

  template<typename prec>
  void gw_qkpt_block<prec>::set_up_qkpt_first(const cxx_complex *V_Qpm_host, int k1, int k2, int q) {
    // this should not trigger. But just in case: wait until we're done with all previous calcs
    cudaStreamSynchronize(stream_);
    k1_ = k1;
    k2_ = k2;
    q_ir_ = q;  // q is index in iBZ

    // block related
    k1_slice_offset_ = symm_utils_.kao_slice_offsets()(k1);
    k2_slice_offset_ = symm_utils_.kao_slice_offsets()(k2);
    k1_slice_size_ = symm_utils_.kao_slice_sizes()(k1);
    k2_slice_size_ = symm_utils_.kao_slice_sizes()(k2);
    V_size_ = symm_utils_.kpair_slice_sizes(k1, k2);
    iq_slice_offset_ = symm_utils_.qaux_slice_offsets_irre()(q);
    iq_slice_size_ = symm_utils_.qaux_slice_sizes_irre()(q);

    std::memcpy(V_Qpm_buffer_, V_Qpm_host,  V_size_ * sizeof(cxx_complex));
    cudaMemcpyAsync(V_Qpm_, V_Qpm_buffer_, V_size_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);

    int two = 2;
    scalar_t alpha = -1;
    cublasSetStream(*handle_, stream_);
    cuda_complex one = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);

    // transpose each block to get V_pmQ
    const tensor<int, 2> &V_irreps = symm_utils_.V_irreps(k1_, k2_);
    const tensor<int, 1> &offsets_V = symm_utils_.V_offsets(k1_, k2_);
    const auto &sizes_k1 = symm_utils_.ao_sizes(k1_);
    const auto &sizes_k2 = symm_utils_.ao_sizes(k2_);
    const auto &sizes_q = symm_utils_.aux_sizes_irre(q_ir_);
    int n_irreps = V_irreps.shape()[0];
    for (int ia = 0; ia < n_irreps; ++ia) {
      int V_offset = offsets_V(ia);
      int iq = V_irreps(ia, 0);
      int ik1 = V_irreps(ia, 1);
      int ik2 = V_irreps(ia, 2);
      int q_size = sizes_q(iq);
      int k1_size = sizes_k1(ik1);
      int k2_size = sizes_k2(ik2);
      // C = alpha*op(A) + beta*op(C)
      if (GEAM(*handle_, CUBLAS_OP_T, CUBLAS_OP_N, q_size, k1_size*k2_size, &one,
               V_Qpm_+V_offset, k1_size*k2_size, &zero,
               V_pmQ_+V_offset, q_size, V_pmQ_+V_offset, q_size) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("GEAM fails on gw_qkpt.set_up_qkpt_first().");
      }
    }

    // conjugate V_Qpm
    //there has to be a better way to compute a complex conjugate!!
    if (RSCAL(*handle_, V_size_, &alpha, (scalar_t *) V_Qpm_ + 1, two) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RSCAL fails on gw_qkpt.set_up_qkpt_first().");
    }

    // set P0 to zero
    cudaMemset(Pqk0_tQP_local_, 0, nt_batch_ * max_qaux_size_ * sizeof(cuda_complex));
    // k_st_ij
    gt_offset_ = ns_ * nt_ * symm_utils_.kao_slice_offsets()(k2_);
    gmt_offset_ = ns_ * nt_ * symm_utils_.kao_slice_offsets()(k1_);
  }

  template<typename prec>
  void gw_qkpt_block<prec>::set_up_qkpt_second(const cxx_complex *V_Qim_host, int k1, int k2, int q) {
    //this should not trigger. But just in case: wait until we're done with all previous calcs
    cudaStreamSynchronize(stream_);
    k1_ = k1;  // k1 is index in full BZ -> need this for getting proper integral information
    k2_ = k2;
    q_ = q;
    k1_ir_ = symm_utils_.irre_index()[k1_];
    q_ir_ = symm_utils_.irre_index()[q_];

    k1_slice_offset_ = symm_utils_.kao_slice_offsets_irre()(k1_ir_);
    k1_slice_size_ = symm_utils_.kao_slice_sizes_irre()(k1_ir_);
    k2_slice_offset_ = symm_utils_.kao_slice_offsets()(k2);
    k2_slice_size_ = symm_utils_.kao_slice_sizes()(k2);
    V_size_ = symm_utils_.kpair_slice_sizes(k1, k2);
    iq_slice_offset_ = symm_utils_.qaux_slice_offsets_irre()(q_ir_);
    iq_slice_size_ = symm_utils_.qaux_slice_sizes_irre()(q_ir_);
    q_slice_size_ = symm_utils_.qaux_slice_sizes()(q_);

    std::memcpy(V_Qpm_buffer_, V_Qim_host, V_size_ * sizeof(cxx_complex));
    cudaMemcpyAsync(V_Qim_, V_Qpm_buffer_, V_size_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);

    // transpose V
    cublasSetStream(*handle_, stream_);
    cuda_complex one = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);

    const tensor<int, 2> &V_irreps = symm_utils_.V_irreps(k1_, k2_);
    const tensor<int, 1> &offsets_V = symm_utils_.V_offsets(k1_, k2_);
    const auto &sizes_k1 = symm_utils_.ao_sizes(k1_);
    const auto &sizes_k2 = symm_utils_.ao_sizes(k2_);
    const auto &sizes_q = symm_utils_.aux_sizes(q_);
    int n_irreps = V_irreps.shape()[0];
    for (int ia = 0; ia < n_irreps; ++ia) {
      int V_offset = offsets_V(ia);
      int iq = V_irreps(ia, 0);
      int ik1 = V_irreps(ia, 1);
      int ik2 = V_irreps(ia, 2);
      int q_size = sizes_q(iq);
      int k1_size = sizes_k1(ik1);
      int k2_size = sizes_k2(ik2);
      // C = alpha*op(A) + beta*op(C)
      if (GEAM(*handle_, CUBLAS_OP_C, CUBLAS_OP_N, q_size*k1_size, k2_size, &one,
               V_Qim_+V_offset, k2_size, &zero,
               V_nPj_+V_offset, q_size*k1_size, V_nPj_+V_offset, q_size*k1_size) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("GEAM fails on gw_qkpt.set_up_qkpt_second().");
      }
    }

    // set sigma to zero
    cudaMemset(sigmak_tsij_, 0, nt_ * ns_ * max_kao_size_ * sizeof(cuda_complex));
    gt_offset_ =  ns_ * nt_ * symm_utils_.kao_slice_offsets()(k2_);
    P_offset_ = nt_ * iq_slice_offset_;
  }

  template<typename prec>
  void gw_qkpt_block<prec>::compute_first_tau_contraction() {

    // ugly but leave it for now ...
    const auto &offsets_k1 = symm_utils_.ao_offsets(k1_);
    const auto &sizes_k1 = symm_utils_.ao_sizes(k1_);
    const auto &offsets_k2 = symm_utils_.ao_offsets(k2_);
    const auto &sizes_k2 = symm_utils_.ao_sizes(k2_);
    const auto &offsets_q = symm_utils_.aux_offsets_irre(q_ir_);
    const auto &sizes_q = symm_utils_.aux_sizes_irre(q_ir_);
    const tensor<int, 2> &V_irreps = symm_utils_.V_irreps(k1_, k2_);
    const tensor<int, 1> &offsets_V = symm_utils_.V_offsets(k1_, k2_);
    int n_irreps = V_irreps.shape()[0];

    cuda_complex one = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    cuda_complex prefactor = (ns_==1) ? cu_type_map<cxx_complex>::cast(-2.,0.) : cu_type_map<cxx_complex>::cast(-1.,0.);
    cublasSetStream(*handle_, stream_);

    // Only compute Pq0(t) for t = [0,beta/2] since Pq0(t) = Pq0(beta-t)
    for (int t = 0; t < nt_ / 2; t += nt_batch_) {
      int nt_mult = std::min(nt_batch_, nt_ / 2 - t);
      for (int s = 0; s < ns_; ++s) {
        int ts = t * ns_ + s;
        //int g_offset = ts * flat_ao_size_;
        for (int ia = 0; ia < n_irreps; ++ia) {
          int iq = V_irreps(ia, 0);
          int ik1 = V_irreps(ia, 1);
          int ik2 = V_irreps(ia, 2);
          int q_size = sizes_q(iq);
          int k1_size = sizes_k1(ik1);
          int k2_size = sizes_k2(ik2);
          int k1_offset = offsets_k1(ik1);
          int k2_offset = offsets_k2(ik2);
          int V_offset = offsets_V(ia);
          int q_offset = offsets_q(iq);

          // contraction at k1
          //X1_t_mQ = G_t_p * V_p_mQ; G_tp = G^{k1}(-t)_tp
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, k2_size*q_size, k1_size, k1_size, &one,
                                   V_pmQ_+V_offset, k2_size*q_size, 0,
                                   g_k_mts_ij_+gmt_offset_+ts*k1_slice_size_+k1_offset, k1_size, ns_*k1_slice_size_,
                                   //g_k_mts_ij_+g_offset+k1_slice_offset_+k1_offset, k1_size, ns_*flat_ao_size_,
                                   &zero, X1t_tmQ_, k2_size*q_size, k1_size*k2_size*q_size,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_first_tau_contraction().");
          }
          // contraction at k2 = k1 + q
          //X2_Pt_m = (V_Pt_n)* G_m_n; G_mn = G^{k2}(t)_{mn}
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_T, CUBLAS_OP_N, k2_size, k1_size*q_size, k2_size, &one,
                                   g_k_ts_ij_+gt_offset_+ts*k2_slice_size_+k2_offset, k2_size, ns_*k2_slice_size_,
                                   //g_k_ts_ij_+g_offset+k2_slice_offset_+k2_offset, k2_size, ns_*flat_ao_size_,
                                   V_Qpm_+V_offset, k2_size, 0,
                                   &zero, X2t_Ptm_, k2_size, k1_size*k2_size*q_size,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_first_tau_contraction().");
          }
          // contraction at q
          //Pq0_QP = X2_Ptm X1_tmQ
          //this should be a +=
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_T, CUBLAS_OP_T, q_size, q_size, k1_size*k2_size, &prefactor,
                                   X2t_Ptm_, k1_size*k2_size, k1_size*k2_size*q_size,
                                   X1t_tmQ_, q_size, k1_size*k2_size*q_size,
                                   &one, Pqk0_tQP_local_+q_offset, q_size, iq_slice_size_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_first_tau_contraction().");
          }
        } // ia
      } // s
      write_P0(t);
      // set P0 to zero
      cudaMemset(Pqk0_tQP_local_, 0, nt_batch_ * max_qaux_size_ * sizeof(cuda_complex));
    } // t
  }

  template<typename prec>
  void gw_qkpt_block<prec>::write_P0(int t) {
    acquire_lock<<<1, 1, 0, stream_>>>(P0_q_locks_ + q_ir_);
    scalar_t one = 1.;
    int nt_mult = std::min(nt_batch_, nt_ / 2 - t);
    /*for (int i = 0; i < nt_mult; ++i) {
      if (RAXPY(*handle_, 2 * iq_slice_size_ * 1, &one, (scalar_t *) (Pqk0_tQP_local_ + i * iq_slice_size_), 1,
                (scalar_t *)(P0_q_t_QP_ + (t+i)*flat_aux_size_ + iq_slice_offset_), 1) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RAXPY fails on gw_qkpt.write_P0().");
      }
    }*/
    if (RAXPY(*handle_, 2 * iq_slice_size_ * nt_mult, &one, (scalar_t *) Pqk0_tQP_local_, 1,
              (scalar_t * )(P0_q_t_QP_ + nt_*iq_slice_offset_ + t*iq_slice_size_), 1) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RAXPY fails on gw_qkpt.write_P0().");
    }
    release_lock<<<1, 1, 0, stream_>>>(P0_q_locks_ + q_ir_);
  }

  template<typename prec>
  void gw_qkpt_block<prec>::get_P(int t) {
    cuda_complex one = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    int two = 2;
    scalar_t alpha = -1;
    int nt_mult = std::min(nt_batch_, nt_-t);
    // transform q_ir -> q
    int cq = symm_utils_.conj_index()[q_];
    if (symm_utils_.index()[cq] == cq) {
      if (cudaMemcpy(Pqk_tQP_local_, P_q_t_QP_+P_offset_+t*iq_slice_size_,
                     nt_mult * iq_slice_size_ * sizeof(cuda_complex),
                     cudaMemcpyDeviceToDevice) != cudaSuccess)
        throw std::runtime_error("failure to copy P from iBZ to fBZ");
    } else {
      const auto &offsets_q = symm_utils_.aux_offsets(cq);
      const auto &sizes_q = symm_utils_.aux_sizes(cq);
      const auto &offsets_iq = symm_utils_.aux_offsets_irre(q_ir_);
      const auto &sizes_iq = symm_utils_.aux_sizes_irre(q_ir_);
      int auxrepslice_offset = symm_utils_.kauxrep_slice_offsets()(cq);
      const auto &irreps = symm_utils_.auxrep_irreps()[cq];
      const auto &offsets = symm_utils_.auxrep_offsets()[cq];
      // orep * dmm * orep.adjoint();
      for (int ia = 0; ia < irreps.shape()[0]; ++ia) {
        int ia1 = irreps(ia, 0);
        int ia2 = irreps(ia, 1);
        int q_size = sizes_q(ia1);
        int iq_size = sizes_iq(ia2);
        int q_offset = offsets_q(ia1);
        int iq_offset = offsets_iq(ia2);
        int o_offset = offsets(ia);
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, iq_size, q_size, iq_size, &one,
                                 P_q_t_QP_+P_offset_+t*iq_slice_size_+iq_offset, iq_size, iq_slice_size_,
                                 kspace_auxrep_+auxrepslice_offset+o_offset, iq_size, 0,
                                 &zero, Pqk0_tQP_local_, iq_size, q_size*iq_size,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("First GEMM_STRIDED_BATCHED fails on transform P.");
        }
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_C, CUBLAS_OP_N, q_size, q_size, iq_size, &one,
                                 kspace_auxrep_+auxrepslice_offset+o_offset, iq_size, 0,
                                 Pqk0_tQP_local_, iq_size, q_size*iq_size,
                                 &zero, Pqk_tQP_local_+q_offset, q_size, q_slice_size_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("Second GEMM_STRIDED_BATCHED fails on transform P.");
        }
      } // ia
    } // rotate
    if (q_ != cq) {
      if (RSCAL(*handle_, nt_mult*q_slice_size_, &alpha,
                (scalar_t *) Pqk_tQP_local_+1, two) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RSCAL fails on transform P.");
      }
    } // conjugate
  }

  template<typename prec>
  void gw_qkpt_block<prec>::compute_second_tau_contraction() {
    // ugly but leave it for now ...
    const auto &offsets_k1 = symm_utils_.ao_offsets(k1_);
    const auto &sizes_k1 = symm_utils_.ao_sizes(k1_);
    const auto &offsets_k2 = symm_utils_.ao_offsets(k2_);
    const auto &sizes_k2 = symm_utils_.ao_sizes(k2_);
    const auto &offsets_q = symm_utils_.aux_offsets(q_);
    const auto &sizes_q = symm_utils_.aux_sizes(q_);
    const tensor<int, 2> &V_irreps = symm_utils_.V_irreps(k1_, k2_);
    const tensor<int, 1> &offsets_V = symm_utils_.V_offsets(k1_, k2_);
    int n_irreps = V_irreps.shape()[0];

    cuda_complex one = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    cuda_complex m1 = cu_type_map<cxx_complex>::cast(-1., 0.);
    cuda_complex *Y1t_Qin = X1t_tmQ_; //name change, reuse memory
    cuda_complex *Y2t_inP = X2t_Ptm_; //name change, reuse memory

    cublasSetStream(*handle_, stream_);
    for (int t = 0; t < nt_; t += nt_batch_) {
      int nt_mult = std::min(nt_batch_, nt_ - t);
      get_P(t);
      for (int s = 0; s < ns_; ++s) {
        int ts = t * ns_ + s;
        int sigma_offset = ts * k1_slice_size_;

        for (int ia = 0; ia < n_irreps; ++ia) {
          int iq = V_irreps(ia, 0);
          int ik1 = V_irreps(ia, 1);
          int ik2 = V_irreps(ia, 2);
          int q_size = sizes_q(iq);
          int k1_size = sizes_k1(ik1);
          int k2_size = sizes_k2(ik2);
          int k1_offset = offsets_k1(ik1);
          int k2_offset = offsets_k2(ik2);
          int V_offset = offsets_V(ia);
          int q_offset = offsets_q(iq);

          // contraction at k2 = k1 - q
          //Y1_Qin = V_Qim * G1_mn; G1_mn = G^{k2}(t)_mn
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, k2_size, k1_size*q_size, k2_size, &one,
                                   g_k_ts_ij_+gt_offset_+ts*k2_slice_size_+k2_offset, k2_size, ns_*k2_slice_size_,
                                   V_Qim_+V_offset, k2_size, 0,
                                   &zero, Y1t_Qin, k2_size, k1_size*k2_size*q_size,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
          }
          //Y2_inP = Y1_Qin * Pq_QP
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, q_size, k1_size*k2_size, q_size, &one,
                                   Pqk_tQP_local_+q_offset, q_size, q_slice_size_,
                                   //P_q_t_QP_+P_offset_+t*q_slice_size_+q_offset, q_size, q_slice_size_,
                                   Y1t_Qin, k1_size*k2_size, k1_size*k2_size*q_size,
                                   &zero, Y2t_inP, q_size, k1_size*k2_size*q_size,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
          }
          //Sigma_ij = Y2_inP V_nPj
          //this should be a +=
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, k1_size, k1_size, k2_size*q_size, &m1,
                                   V_nPj_+V_offset, k1_size, 0,
                                   Y2t_inP, k2_size*q_size, k1_size*k2_size*q_size,
                                   &one, sigmak_tsij_+sigma_offset+k1_offset, k1_size, ns_ * k1_slice_size_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
          }
        } // ia
      } // s
    } // t
    write_sigma();
  }

  template<typename prec>
  void gw_qkpt_block<prec>::write_sigma(){
    //write results. Make sure we have exclusive write access to sigma, then add array sigmak_tij to sigma_ktij
    acquire_lock<<<1, 1, 0, stream_>>>(sigma_k_locks_ + k1_);
    scalar_t one = 1.;

    if (RAXPY(*handle_, 2 * nt_ * ns_ * k1_slice_size_, &one, (scalar_t *) sigmak_tsij_, 1,
              (scalar_t *) (sigma_k_ts_ij_ + nt_*ns_*k1_slice_offset_), 1) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RAXPY fails on gw_qkpt.write_sigma().");
    }
    /*for (int t = 0; t < nt_; ++t) {
      for (int s = 0; s < ns_; ++s) {
        int ts = t * ns_ + s;
        int tsij_shift = ts * k1_slice_size_;
        int tskij_shift = ts * flat_ao_size_ + k1_slice_offset_;
        if (RAXPY(*handle_, 2 * k1_slice_size_, &one, (scalar_t *) (sigmak_tsij_ + tsij_shift), 1,
                  (scalar_t *) (sigma_k_ts_ij_ + tskij_shift), 1) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("RAXPY fails on gw_qkpt.write_sigma().");
        }
      }
    }*/
    release_lock<<<1, 1, 0, stream_>>>(sigma_k_locks_ + k1_);
  }

  template<typename prec>
  bool gw_qkpt_block<prec>::is_busy() {
    cudaError_t stream_status = cudaStreamQuery(stream_);
    if (stream_status == cudaSuccess) return false; //not busy;
    else if (stream_status == cudaErrorNotReady) return true; //busy~
    else throw std::runtime_error("problem with stream query");
  }

  template<typename prec>
  void gw_qkpt_block<prec>::synchronize() {
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
      throw std::runtime_error("could not wait for other streams");
  }

  template class gw_qkpt_block<float>;
  template class gw_qkpt_block<double>;

} // namespace symmetry_mbpt

