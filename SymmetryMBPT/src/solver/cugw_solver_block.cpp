
#include <cuda_profiler_api.h>

#include "analysis.h"
#include "cugw_solver_block.h"

namespace symmetry_mbpt {

  void cugw_solver_block_t::GW_complexity_estimation() {
    flop_count_ = GW_complexity_estimation_block(world_rank_, symm_utils_, ns_, nts_, nw_b_, symm_utils_.ink());
  }

  void cugw_solver_block_t::allocate_shared_memory_P() {
    MPI_Aint size;
    if (intranode_rank_ == 0) {
      size = nts_ * symm_utils_.flat_aux_size() * sizeof(dcomplex);
      if (!world_rank_) {
        std::cout << "Node-wide shared memory allocation of polarization: "
                  << size / 1024. / 1024. << " MB" << std::endl;
      }
      MPI_Win_allocate_shared(size, sizeof(dcomplex), MPI_INFO_NULL, intranode_comm_, &P0_data_, &win_P_);
    }
    else {
      int disp_unit;
      MPI_Win_allocate_shared(0, sizeof(dcomplex), MPI_INFO_NULL, intranode_comm_, &P0_data_, &win_P_);
      MPI_Win_shared_query(win_P_, 0, &size, &disp_unit, &P0_data_);
    }
    P0_q_t_QP_host_.set_ref(P0_data_);
    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) P0_q_t_QP_host_.set_zero();
    MPI_Win_fence(0, win_P_);
  }

  void cugw_solver_block_t::solve() {

    gw_statistics.start("total");
    setup_solve();

    if (iter_ < 3) { cudaProfilerStart(); }
    // Only those processes assigned with a device will be involved in GW self-energy calculation
    gw_statistics.start("GW_loop");
    //if (devices_comm_ != MPI_COMM_NULL) {
    gw_innerloop();
    //}
    MPI_Barrier(world_comm_);
    gw_statistics.end("GW_loop");
    MPI_Win_fence(0, win_Sigma_);
    if (!intranode_rank_) {
      if (devices_comm_ != MPI_COMM_NULL) gw_statistics.start("selfenergy_reduce");
      MPI_Allreduce(MPI_IN_PLACE, Sigma_ts_kij_host_.data(), Sigma_ts_kij_host_.size(),
                    MPI_C_DOUBLE_COMPLEX, MPI_SUM, internode_comm_);
      Sigma_ts_kij_host_ /= (nk_);
      if (devices_comm_ != MPI_COMM_NULL) gw_statistics.end("selfenergy_reduce");
    }
    MPI_Win_fence(0, win_Sigma_);
    MPI_Barrier(world_comm_);
    gw_statistics.end("total");
    gw_statistics.print(world_comm_);

    clean_MPI_structure();
    delete coul_int_;
    iter_ += 1;
    MPI_Barrier(world_comm_);
    if (iter_ < 3) { cudaProfilerStop(); }
  }

  void cugw_solver_block_t::solve_P0() {
    setup_solve();
    gw_statistics.start("P0_loop");

    if (!sp_) {
      compute_gw_P0<double>();
    } else {
      compute_gw_P0<float>();
    }

    MPI_Barrier(world_comm_);
    gw_statistics.end("P0_loop");
    gw_statistics.print(world_comm_);

    clean_MPI_structure();
    delete coul_int_;
    MPI_Barrier(world_comm_);
  }

  void cugw_solver_block_t::solve_sigma() {
    setup_solve();
    gw_statistics.start("sigma_loop");

    if (!sp_) {
      compute_gw_selfenergy_P0<double>();
    } else {
      compute_gw_selfenergy_P0<float>();
    }

    MPI_Win_fence(0, win_Sigma_);
    if (!intranode_rank_) {
      if (devices_comm_ != MPI_COMM_NULL) gw_statistics.start("selfenergy_reduce");
      MPI_Allreduce(MPI_IN_PLACE, Sigma_ts_kij_host_.data(), Sigma_ts_kij_host_.size(),
                    MPI_C_DOUBLE_COMPLEX, MPI_SUM, internode_comm_);
      Sigma_ts_kij_host_ /= (nk_);
      if (devices_comm_ != MPI_COMM_NULL) gw_statistics.end("selfenergy_reduce");
    }
    MPI_Win_fence(0, win_Sigma_);
    MPI_Barrier(world_comm_);
    gw_statistics.end("sigma_loop");
    gw_statistics.print(world_comm_);

    clean_MPI_structure();
    delete coul_int_;
    MPI_Barrier(world_comm_);
  }

  void cugw_solver_block_t::setup_solve() {
    gw_statistics.start("Initialization");
    setup_MPI_structure();
    MPI_Win_fence(0, win_Sigma_);
    if (!intranode_rank_) Sigma_ts_kij_host_.set_zero();
    MPI_Win_fence(0, win_Sigma_);

    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) P0_q_t_QP_host_.set_zero();
    MPI_Win_fence(0, win_P_);

    coul_int_ = new df_integral_block_t(path_, symm_utils_, "flat_");
    MPI_Barrier(world_comm_);
    gw_statistics.end("Initialization");
    update_integrals(coul_int_, gw_statistics);
  }

  void cugw_solver_block_t::gw_innerloop() {
    if (!sp_) {
      compute_gw_selfenergy<double>();
    } else {
      compute_gw_selfenergy<float>();
    }
  }

  template<typename prec>
  void cugw_solver_block_t::compute_gw_P0() {

    // It should be fine to have these two definitions on all ranks even though they are not used
    mom_cons_callback mom_cons = [&](const std::array<size_t, 3> &k123) -> std::array<size_t, 4> {
      return symm_utils_.momentum_conservation(k123);
    };

    integral_reader_callback<prec> reader = [&](int k, int k1, tensor<std::complex<prec>, 1> &V_Qpm,
                                                std::complex<double> *Vk1k2_Qij) {
      gw_statistics.start("read");
      coul_int_->symmetrize(Vk1k2_Qij, V_Qpm, k, k1);
      gw_statistics.end("read");
    };

    std::unique_ptr<cugw_routine_block<prec> > cugw_ptr = nullptr;

    // Only those processes assigned with a device will be involved in P0 and Sigma calculation
    if (devices_comm_ != MPI_COMM_NULL) {
      // check devices' free space and space requirements
      // this set nqkpt_ to a proper value
      GW_check_devices_free_space();

      gw_statistics.start("Initialization");
      cugw_ptr = std::make_unique<cugw_routine_block<prec> >(nk_, ink_, ns_, nt_batch_, nts_,
                                                             symm_utils_.k_pairs(), symm_utils_,
                                                             Vk1k2_Qij_,
                                                             nqkpt_,
                                                             devices_rank_, devices_size_,
                                                             world_rank_, intranode_rank_, devCount_per_node_);
      gw_statistics.end("Initialization");

      gw_statistics.start("Copy data");
      cugw_ptr->copy_ORep_host_to_device(symm_utils_.kspace_orep().data());
      cugw_ptr->copy_G_host_to_device(G_ts_kij_host_.data());
      // TODO: debug only. copy G back and compare. remove this
      //cugw_ptr->copy_G_device_to_host(G_kstij_host_.data());
      cugw_ptr->copy_AuxRep_host_to_device(symm_utils_.kspace_auxrep().data());
      gw_statistics.end("Copy data");

      gw_statistics.start("Compute P0");
      compute_P0q(cugw_ptr, mom_cons, reader);
      gw_statistics.end("Compute P0");
    }

    // these two steps are performed on all processes
    reduce_Pq();
    // divide nk prefactor. TODO: check if this is correct
    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) P0_q_t_QP_host_ /= nk_;
    MPI_Win_fence(0, win_P_);
  }

  template<typename prec>
  void cugw_solver_block_t::compute_gw_selfenergy_P0() {
    // It should be fine to have these two definitions on all ranks even though they are not used
    mom_cons_callback mom_cons = [&](const std::array<size_t, 3> &k123) -> std::array<size_t, 4> {
      return symm_utils_.momentum_conservation(k123);
    };

    integral_reader_callback<prec> reader = [&](int k, int k1, tensor<std::complex<prec>, 1> &V_Qpm,
                                                std::complex<double> *Vk1k2_Qij) {
      gw_statistics.start("read");
      coul_int_->symmetrize(Vk1k2_Qij, V_Qpm, k, k1);
      gw_statistics.end("read");
    };

    std::unique_ptr<cugw_routine_block<prec> > cugw_ptr = nullptr;

    // Only those processes assigned with a device will be involved in P0 and Sigma calculation
    if (devices_comm_ != MPI_COMM_NULL) {
      // check devices' free space and space requirements
      // this set nqkpt_ to a proper value
      GW_check_devices_free_space();

      gw_statistics.start("Initialization");
      cugw_ptr = std::make_unique<cugw_routine_block<prec> >(nk_, ink_, ns_, nt_batch_, nts_,
                                                             symm_utils_.k_pairs(), symm_utils_,
                                                             Vk1k2_Qij_,
                                                             nqkpt_,
                                                             devices_rank_, devices_size_,
                                                             world_rank_, intranode_rank_, devCount_per_node_);
      gw_statistics.end("Initialization");

      gw_statistics.start("Copy data");
      cugw_ptr->copy_ORep_host_to_device(symm_utils_.kspace_orep().data());
      cugw_ptr->copy_G_host_to_device(G_ts_kij_host_.data());
      // TODO: debug only. copy G back and compare. remove this
      //cugw_ptr->copy_G_device_to_host(G_kstij_host_.data());
      cugw_ptr->copy_AuxRep_host_to_device(symm_utils_.kspace_auxrep().data());
      gw_statistics.end("Copy data");

      gw_statistics.start("Compute P0");
      compute_P0q(cugw_ptr, mom_cons, reader);
      gw_statistics.end("Compute P0");
    }

    // these two steps are performed on all processes
    reduce_Pq();
    // divide nk prefactor. TODO: check if this is correct
    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) P0_q_t_QP_host_ /= nk_;
    MPI_Win_fence(0, win_P_);

    gw_statistics.start("Compute full P0");
    obtain_full_P0q();
    gw_statistics.end("Compute full P0");

    if (devices_comm_ != MPI_COMM_NULL) {
      gw_statistics.start("Compute Sigma");
      compute_Sigmak(cugw_ptr, mom_cons, reader);
      gw_statistics.end("Compute Sigma");
    }
  }

  template<typename prec>
  void cugw_solver_block_t::compute_gw_selfenergy() {

    // It should be fine to have these two definitions on all ranks even though they are not used
    mom_cons_callback mom_cons = [&](const std::array<size_t, 3> &k123) -> std::array<size_t, 4> {
      return symm_utils_.momentum_conservation(k123);
    };

    integral_reader_callback<prec> reader = [&](int k, int k1, tensor<std::complex<prec>, 1> &V_Qpm,
                                                std::complex<double> *Vk1k2_Qij) {
      gw_statistics.start("read");
      coul_int_->symmetrize(Vk1k2_Qij, V_Qpm, k, k1);
      gw_statistics.end("read");
    };

    std::unique_ptr<cugw_routine_block<prec> > cugw_ptr = nullptr;

    // Only those processes assigned with a device will be involved in P0 and Sigma calculation
    if (devices_comm_ != MPI_COMM_NULL) {
      // check devices' free space and space requirements
      // this set nqkpt_ to a proper value
      GW_check_devices_free_space();

      gw_statistics.start("Initialization");
      cugw_ptr = std::make_unique<cugw_routine_block<prec> >(nk_, ink_, ns_, nt_batch_, nts_,
                                                             symm_utils_.k_pairs(), symm_utils_,
                                                             Vk1k2_Qij_,
                                                             nqkpt_,
                                                             devices_rank_, devices_size_,
                                                             world_rank_, intranode_rank_, devCount_per_node_);
      gw_statistics.end("Initialization");

      gw_statistics.start("Copy data");
      cugw_ptr->copy_ORep_host_to_device(symm_utils_.kspace_orep().data());
      cugw_ptr->copy_G_host_to_device(G_ts_kij_host_.data());
      // TODO: debug only. copy G back and compare. remove this
      //cugw_ptr->copy_G_device_to_host(G_kstij_host_.data());
      cugw_ptr->copy_AuxRep_host_to_device(symm_utils_.kspace_auxrep().data());
      gw_statistics.end("Copy data");

      gw_statistics.start("Compute P0");
      compute_P0q(cugw_ptr, mom_cons, reader);
      gw_statistics.end("Compute P0");
    }

    // these two steps are performed on all processes
    reduce_Pq();
    // divide nk prefactor. TODO: check if this is correct
    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) P0_q_t_QP_host_ /= nk_;
    MPI_Win_fence(0, win_P_);

    gw_statistics.start("Compute P");
    compute_Pq_from_P0q();
    gw_statistics.end("Compute P");

    if (devices_comm_ != MPI_COMM_NULL) {
      gw_statistics.start("Compute Sigma");
      compute_Sigmak(cugw_ptr, mom_cons, reader);
      gw_statistics.end("Compute Sigma");
    }
  }

  template<typename prec>
  void cugw_solver_block_t::compute_P0q(std::unique_ptr<cugw_routine_block<prec> > &cugw_ptr,
                                        mom_cons_callback &mom_cons, integral_reader_callback<prec> &reader) {
    // Since all process in _devices_comm will write to the P0 simultaneously,
    // instead of adding locks in cugw.solve(), we locate private P0_local_host
    // and do MPIAllreduce on CPU later on. Since the number of processes with a GPU is very
    // limited, the additional memory overhead is fairly limited.
    tensor<dcomplex, 1> P0_q_t_QP_host_local(nts_*symm_utils_.flat_aux_size());
    P0_q_t_QP_host_local.set_zero(); // just to be safe. default of alps tensor should be zero

    gw_statistics.start("P0_contraction");
    cugw_ptr->solve_P0(mom_cons, reader);
    gw_statistics.end("P0_contraction");
    cugw_ptr->copy_P0_device_to_host(P0_q_t_QP_host_local.data());

    // Copy back to P0_tqQP_host
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_P_);
    P0_q_t_QP_host_ += P0_q_t_QP_host_local;
    MPI_Win_unlock(0, win_P_);
  }

  template<typename prec>
  void cugw_solver_block_t::compute_Sigmak(std::unique_ptr<cugw_routine_block<prec> > &cugw_ptr,
                                           mom_cons_callback &mom_cons, integral_reader_callback<prec> &reader) {
    tensor<dcomplex, 3> Sigma_ts_kij_host_local(nts_, ns_, symm_utils_.flat_ao_size());
    Sigma_ts_kij_host_local.set_zero();

    cugw_ptr->copy_P_host_to_device(P0_q_t_QP_host_.data());
    gw_statistics.start("Sigma_contraction");
    cugw_ptr->solve_Sigma(mom_cons, reader);
    gw_statistics.end("Sigma_contraction");
    cugw_ptr->copy_Sigma_device_to_host(Sigma_ts_kij_host_local.data());

    // Copy back to Sigma_tskij_local_host
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_Sigma_);
    Sigma_ts_kij_host_ += Sigma_ts_kij_host_local;
    MPI_Win_unlock(0, win_Sigma_);
  }

  // Explicit instatiations
  template void cugw_solver_block_t::compute_gw_P0<float>();

  template void cugw_solver_block_t::compute_gw_P0<double>();

  template void cugw_solver_block_t::compute_gw_selfenergy_P0<float>();

  template void cugw_solver_block_t::compute_gw_selfenergy_P0<double>();

  template void cugw_solver_block_t::compute_gw_selfenergy<float>();

  template void cugw_solver_block_t::compute_gw_selfenergy<double>();

  template void cugw_solver_block_t::compute_P0q<float>(std::unique_ptr<cugw_routine_block<float> > &,
                                                        mom_cons_callback &,
                                                        integral_reader_callback<float> &);

  template void cugw_solver_block_t::compute_P0q<double>(std::unique_ptr<cugw_routine_block<double> > &,
                                                         mom_cons_callback &,
                                                         integral_reader_callback<double> &);

  template void cugw_solver_block_t::compute_Sigmak<float>(std::unique_ptr<cugw_routine_block<float> > &,
                                                           mom_cons_callback &,
                                                           integral_reader_callback<float> &);

  template void cugw_solver_block_t::compute_Sigmak<double>(std::unique_ptr<cugw_routine_block<double> > &,
                                                            mom_cons_callback &,
                                                            integral_reader_callback<double> &);

  void cugw_solver_block_t::compute_Pq_from_P0q() {

    for (size_t iq = world_rank_; iq < ink_; iq += world_size_) {
      // TODO: temporarily allocate inside loop so processes not involved in solving P will not allocate memory.
      //  Ideally we want to split the allocation over nodes

      int qslice_size = symm_utils_.qaux_slice_sizes_irre()(iq);
      int qslice_offset = symm_utils_.qaux_slice_offsets_irre()(iq);
      tensor<dcomplex, 2> P0_w(nw_b_, qslice_size);
      tensor<dcomplex, 2> P0_t(nts_, qslice_size);
      for (int t = 0; t < nts_ / 2; ++t) {
        P0_t(t).vector() = CMcolumn<dcomplex>(P0_q_t_QP_host_.data() + nts_*qslice_offset + t*qslice_size, qslice_size);
        P0_t(nts_ - t - 1).vector() = P0_t(t).vector();
      }
      // Transform P0_tilde from Fermionic tau to Bonsonic Matsubara grid
      ft_.tau_f_to_w_b(P0_t, P0_w);
      // TODO: print leakage

      // loop over all blocks for a single iq point
      int n_block = symm_utils_.aux_sizes_irre(iq).size();
      const auto &offsets_q = symm_utils_.aux_offsets_irre(iq);
      const auto &sizes_q = symm_utils_.aux_sizes_irre(iq);

      // Solve Dyson-like eqn for nw_b frequency points
      MatrixXcd identity;
      //Eigen::FullPivLU<MatrixXcd> lusolver;
      Eigen::LDLT<MatrixXcd> ldltsolver;
      for (size_t n = 0; n < nw_b_; ++n) {
        for (int ia = 0; ia < n_block; ++ia) {
          int q_size = sizes_q(ia);
          int q_offset = offsets_q(ia);
          identity = MatrixX<dcomplex>::Identity(q_size, q_size);
          MatrixX<dcomplex> temp = MMatrixX<dcomplex>(P0_w(n).data() + q_offset, q_size, q_size);
          temp = identity - 0.5 * (temp + temp.conjugate().transpose().eval());
          //temp = lusolver.compute(temp).inverse().eval();
          temp = ldltsolver.compute(temp).solve(identity).eval();
          temp = 0.5 * (temp + temp.conjugate().transpose().eval());
          MMatrixX<dcomplex>(P0_w(n).data() + q_offset, q_size, q_size)
              = (temp * MMatrixX<dcomplex>(P0_w(n).data() + q_offset, q_size, q_size)).eval();
        }
      }

      // Transform back from Bosonic Matsubara to Fermionic tau.
      ft_.w_b_to_tau_f(P0_w, P0_t);

      // TODO: is this the right way of doing it in shared memory?
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_P_);
      for (int t = 0; t < nts_; ++t) {
        Mcolumn<dcomplex>(P0_q_t_QP_host_.data() + nts_*qslice_offset + t*qslice_size, qslice_size) = P0_t(t).vector();
      }
      MPI_Win_unlock(0, win_P_);
      //TODO: print leakage
    }
    reduce_Pq();
  }

  void cugw_solver_block_t::obtain_full_P0q() {
    MPI_Barrier(world_comm_);
    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) {
      for (int iq = 0; iq < ink_; ++iq) {
        int qslice_size = symm_utils_.qaux_slice_sizes_irre()(iq);
        int qslice_offset = symm_utils_.qaux_slice_offsets_irre()(iq);
        for (int t = 0; t < nts_ / 2; ++t) {
          Mcolumn<dcomplex>(P0_q_t_QP_host_.data() + nts_*qslice_offset + (nts_-t-1)*qslice_size, qslice_size)
              = Mcolumn<dcomplex>(P0_q_t_QP_host_.data() + nts_*qslice_offset + t*qslice_size, qslice_size);
        }
      }
    }
    MPI_Win_fence(0, win_P_);
    MPI_Barrier(world_comm_);
  }

  void cugw_solver_block_t::reduce_Pq() {
    MPI_Barrier(world_comm_);
    MPI_Win_fence(0, win_P_);
    if (!intranode_rank_) {
      gw_statistics.start("P_reduce");
      MPI_Allreduce(MPI_IN_PLACE, P0_q_t_QP_host_.data(), P0_q_t_QP_host_.size(),
                    MPI_C_DOUBLE_COMPLEX, MPI_SUM, internode_comm_);
      gw_statistics.end("P_reduce");
    }
    MPI_Win_fence(0, win_P_);
    MPI_Barrier(world_comm_);
  }

  void cugw_solver_block_t::GW_check_devices_free_space() {
    // check devices' free space and space requirements
    std::cout << std::setprecision(4) << std::boolalpha;

    if (!devices_rank_) std::cout << "size of tau batch: " << nt_batch_ << std::endl;

    std::size_t qkpt_size = (!sp_) ? gw_qkpt_block<double>::size(max_kao_size_, max_qaux_size_, max_V_size_, nts_, nt_batch_, ns_)
                                   : gw_qkpt_block<float>::size(max_kao_size_, max_qaux_size_, max_V_size_, nts_, nt_batch_, ns_);
    std::size_t available_memory;
    std::size_t total_memory;
    cudaMemGetInfo(&available_memory, &total_memory);
    nqkpt_ = std::min(int((available_memory * 0.8) / qkpt_size), 16);
    if (!devices_rank_) {
      std::cout << "size per qkpt: " << qkpt_size / (1024. * 1024. * 1024.) << " GB " << std::endl;
      std::cout << "available memory: " << available_memory / (1024. * 1024. * 1024.) << " GB " << " of total: "
                << total_memory / (1024. * 1024. * 1024.) << " GB. " << std::endl;
      std::cout << "can create: " << nqkpt_ << " qkpts in parallel" << std::endl;
    }
    if (nqkpt_ == 0) throw std::runtime_error("not enough memory to create qkpt. Please reduce nt_batch");
    if (nqkpt_ == 1 && !world_rank_)
      std::cerr << "WARNING: ONLY ONE QKPT CREATED. LIKELY CODE WILL BE SLOW. REDUCE NT_BATCH" << std::endl;
    std::cout << std::setprecision(15);
  }

} // namespace symmetry_mbpt