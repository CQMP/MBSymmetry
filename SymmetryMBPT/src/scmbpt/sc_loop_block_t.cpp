
#include "sc_loop_block_t.h"

namespace symmetry_mbpt{

  void sc_loop_block_t::initialize_shmem_communicators() {
    //shared memory communicator
    MPI_Comm_split_type(world_comm_, MPI_COMM_TYPE_SHARED, world_rank_, MPI_INFO_NULL, &node_comm_);
    MPI_Comm_size(node_comm_, &node_size_);
    MPI_Comm_rank(node_comm_, &node_rank_);

    //spliting into an inter-node communicator (note that we'll only ever use node_rank zero for this
    int color = world_rank_ % node_size_;
    int key = world_rank_ / node_size_;
    MPI_Comm_split(world_comm_, color, key, &internode_comm_);
    MPI_Comm_size(internode_comm_, &internode_size_);
    MPI_Comm_rank(internode_comm_, &internode_rank_);

    if (!world_rank_) {
      std::cout << "********************************" << std::endl;
      std::cout << "***** MPI Information **********" << std::endl;
      std::cout << "***** size                 : " << world_size_ << std::endl;
      std::cout << "***** node, i.e. shmem size: " << node_size_ << std::endl;
      std::cout << "********************************" << std::endl;
    }

    solver_->solver_block_t::set_intranode_MPI(node_comm_, node_rank_, node_size_);
    solver_->solver_block_t::set_internode_MPI(internode_comm_, internode_rank_, internode_size_);
  }

  void sc_loop_block_t::allocate_shared_memory() {
    MPI_Aint size;
    if (node_rank_ == 0) {
      size = nts_ * ns_ * symm_utils_.flat_ao_size() * sizeof(std::complex<double>);
      if (!world_rank_)
        std::cout << "Node-wide shared memory allocation of Green's function and self-energy: "
                  << 2 * size / 1024. / 1024. << " MB" << std::endl;
      MPI_Win_allocate_shared(size, sizeof(std::complex<double>), MPI_INFO_NULL, node_comm_, &G_tau_data_, &win_G_);
      MPI_Win_allocate_shared(size, sizeof(std::complex<double>), MPI_INFO_NULL, node_comm_, &Sigma_tau_data_,
                              &win_Sigma_);
    } else {
      int disp_unit;
      MPI_Win_allocate_shared(0, sizeof(std::complex<double>), MPI_INFO_NULL, node_comm_, &G_tau_data_, &win_G_);
      MPI_Win_allocate_shared(0, sizeof(std::complex<double>), MPI_INFO_NULL, node_comm_, &Sigma_tau_data_,
                              &win_Sigma_);
      MPI_Win_shared_query(win_G_, 0, &size, &disp_unit, &G_tau_data_);
      MPI_Win_shared_query(win_Sigma_, 0, &size, &disp_unit, &Sigma_tau_data_);
    }
    solver_->solver_block_t::set_Sigma_MPI_window(win_Sigma_);
    //diis_.set_G_Sigma_MPI_window(win_Sigma_);
    G_tau_.set_ref(G_tau_data_);
    Selfenergy_.set_ref(Sigma_tau_data_);
    MPI_Win_fence(0, win_G_);
    if (!node_rank_) G_tau_.set_zero();
    MPI_Win_fence(0, win_G_);
    MPI_Win_fence(0, win_Sigma_);
    if (!node_rank_) Selfenergy_.set_zero();
    MPI_Win_fence(0, win_Sigma_);

    solver_->set_shared_Coulomb();
    solver_->allocate_solver_shared_memory();
  }

  void sc_loop_block_t::read_input_data(const std::string &path) {

    tensor<dcomplex, 2> F_k_tmp(ns_, symm_utils_.flat_ao_size_full());
    tensor<dcomplex, 2> S_k_tmp(ns_, symm_utils_.flat_ao_size_full());
    tensor<dcomplex, 2> H_k_tmp(ns_, symm_utils_.flat_ao_size_full());
    alps::hdf5::archive in_file(path.c_str(), "r");
    in_file["HF/Fock-k"] >> F_k_tmp;
    in_file["HF/S-k"] >> S_k_tmp;
    in_file["HF/H-k"] >> H_k_tmp;
    in_file["HF/madelung"] >> madelung_;
    in_file["HF/Energy_nuc"] >> enuc_;
    in_file["HF/Energy"] >> ehf_;
    in_file.close();

    // F, S, K store reduced k points
    for (int is = 0; is < ns_; ++is) {
      for (size_t ik = 0; ik < ink_; ++ik) {
        size_t k = symm_utils_.irre_list()[ik];
        int ik_slice_offset = symm_utils_.kao_slice_offsets_irre()(ik);
        int k_slice_offset = symm_utils_.kao_slice_offsets()(k);
        int ik_slice_size = symm_utils_.kao_slice_sizes_irre()(ik);
        int k_slice_size = symm_utils_.kao_slice_sizes()(k);
        Mcolumn<dcomplex>(F_k_(is).data()+ik_slice_offset, ik_slice_size)
            = Mcolumn<dcomplex>(F_k_tmp(is).data()+k_slice_offset, k_slice_size);
        Mcolumn<dcomplex>(H_k_(is).data()+ik_slice_offset, ik_slice_size)
            = Mcolumn<dcomplex>(H_k_tmp(is).data()+k_slice_offset, k_slice_size);
        Mcolumn<dcomplex>(S_k_(is).data()+ik_slice_offset, ik_slice_size)
            = Mcolumn<dcomplex>(S_k_tmp(is).data()+k_slice_offset, k_slice_size);
      }
    }

    F_k_ -= H_k_;
    for (int is = 0; is < ns_; ++is) {
      H_k_(is) += S_k_(is) * Bz_ * (!is ? 0.5 : -0.5);
    }
    F_k_ += H_k_;

    make_hermitian_kao(F_k_, symm_utils_);
    make_hermitian_kao(S_k_, symm_utils_);
    make_hermitian_kao(H_k_, symm_utils_);

    if (ink_ != symm_utils_.ink()) {
      std::cerr << "ink_: " << ink_ << " should be: " << symm_utils_.ink() << std::endl;
      throw std::logic_error("ink_ doesn't match with symmetry utils");
    }
  }

} // namespace symmetry_mbpt