#ifndef SYMMETRYMBPT_SOLVER_BLOCK_T_H
#define SYMMETRYMBPT_SOLVER_BLOCK_T_H

#include <mpi.h>

#include "type.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

  class solver_block_t {
  public:
    solver_block_t(MPI_Comm comm, const symmetry_utils_t &symm_utils, int ns, const std::string &dfintegral_hf_file) :
        world_comm_(comm), symm_utils_(symm_utils),
        nk_(symm_utils.nk()), ink_(symm_utils.ink()), ns_(ns),
        hf_path_(dfintegral_hf_file) {
      MPI_Comm_size(world_comm_, &world_size_);
      MPI_Comm_rank(world_comm_, &world_rank_);
    }

    virtual ~solver_block_t() = default;

    void set_internode_MPI(MPI_Comm internode_comm, int internode_rank, int internode_size) {
      internode_comm_ = internode_comm;
      internode_rank_ = internode_rank;
      internode_size_ = internode_size;
    }
    void set_intranode_MPI(MPI_Comm intranode_comm, int intranode_rank, int intranode_size) {
      intranode_comm_ = intranode_comm;
      intranode_rank_ = intranode_rank;
      intranode_size_ = intranode_size;
    }
    void set_Sigma_MPI_window(MPI_Win win_Sigma) {
      win_Sigma_ = win_Sigma;
    }

    virtual void solve() = 0;

    virtual tensor<dcomplex, 2> solve_HF(const tensor<dcomplex, 2>&, const tensor<dcomplex, 2>&,
                                         const tensor<dcomplex, 2>&, double madelung) = 0;

    virtual void get_iter(int &iter) = 0;

    /**
     * Determine whether anything will be executed in single precision
     * @param run_sp - [INPUT] Whether to switch to single precision
     */
    virtual void single_prec_run(bool run_sp) = 0;

    /**
     * Dummy function. Will be overwritten in cuda solver at high cpu memory mode
     */
    virtual void set_shared_Coulomb() = 0;

    virtual void allocate_solver_shared_memory() {};

    virtual std::string name() const = 0;

    /*
     * GW test related
     */
    virtual const tensor_view<dcomplex, 1> &P0_q_t_QP() const { throw std::runtime_error("P0_qtQP not implemented"); }
    virtual void solve_P0() { throw std::runtime_error("solve P0 not implemented"); }
    virtual void solve_sigma() { throw std::runtime_error("solve sigma not implemented"); }
    virtual void compute_Pq_from_P0q() { throw std::runtime_error("compute Pq from P0q not implemented"); }

  protected:

    // MPI Communicator to be used
    MPI_Comm world_comm_;
    int world_size_;
    int world_rank_;
    MPI_Comm internode_comm_;
    int internode_rank_;
    int internode_size_;
    MPI_Comm intranode_comm_;
    int intranode_rank_;
    int intranode_size_;
    MPI_Win win_Sigma_;

    // number of cells for GF2 loop
    size_t nk_;
    // number of k-point after time-reversal symmetry
    size_t ink_;
    // number of spins
    size_t ns_;

    const symmetry_utils_t &symm_utils_;

    const std::string hf_path_;
  };

} // namespace symmetry_mbpt


#endif //SYMMETRYMBPT_SOLVER_BLOCK_T_H
