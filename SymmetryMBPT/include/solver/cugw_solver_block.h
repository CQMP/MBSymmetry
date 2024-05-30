#ifndef SYMMETRYMBPT_CUGW_SOLVER_BLOCK_H
#define SYMMETRYMBPT_CUGW_SOLVER_BLOCK_H

#include "transformer_t.h"
#include "cuhf_solver_block.h"
#include "cugw_routines_block.h"

namespace symmetry_mbpt {

  class cugw_solver_block_t : public cuhf_solver_block_t {
  public:
    /**
     * Class constructor
     *
     * @param comm       -- global communicator
     * @param ft         -- Fourier transformer between imaginary time and frequency axis
     * @param symm_utils -- symmetry utilities
     * @param Gk         -- Lattice Green's function (tau, ns, ink, nao, nao)
     * @param Sigma      -- Lattice self-energy (tau, ns, ink, nao, nao)
     */
    cugw_solver_block_t(MPI_Comm comm, const params_t &p, const transformer_t &ft, const symmetry_utils_t &symm_utils,
                        tensor_view<dcomplex, 3> &Gk, tensor_view<dcomplex, 3> &Sigmak):
        cugw_solver_block_t(comm, ft, symm_utils, Gk, Sigmak, p.dfintegral_hf_file, p.dfintegral_file,
                            p.ns, p.ni+2, p.nt_batch) {}
    cugw_solver_block_t(MPI_Comm comm, const transformer_t &ft, const symmetry_utils_t &symm_utils,
                        tensor_view<dcomplex, 3> &Gk, tensor_view<dcomplex, 3> &Sigmak,
                        const std::string &dfintegral_hf_file, const std::string &dfintegral_file,
                        int ns, int nts, int nt_batch, size_t verbose=1):
                        cuhf_solver_block_t(comm, symm_utils, ns, dfintegral_hf_file), ft_(ft),
                        max_kao_size_(symm_utils.max_kao_size()), max_qaux_size_(symm_utils.max_qaux_size()),
                        max_V_size_(symm_utils.max_kpair_size()),
                        nts_(nts), nt_batch_(nt_batch),
                        nw_b_(ft.wsample_bose().size()),
                        path_(dfintegral_file),
                        sp_(false), nqkpt_(0), flop_count_(0), iter_(1),
                        G_ts_kij_host_(Gk), Sigma_ts_kij_host_(Sigmak),
                        P0_q_t_QP_host_(nullptr, nts_*symm_utils.flat_aux_size()) {
      // Check if nts is an even number since we will take the advantage of Pq0(beta-t) = Pq0(t) later
      if(nts_ % 2 != 0)
        throw std::runtime_error("Number of tau points should be even number");

      if (verbose > 0) {
        GW_complexity_estimation();
      }
      init_events();
    }

    ~cugw_solver_block_t() override {
      deallocate_shared_memory_P();
    };

    /**
     * Solve GW self-energy
     */
    void solve() override;

    // only compute P0, for testing purpose only
    void solve_P0() override;
    // compute sigma_gw with P0, for testing purpose only
    void solve_sigma() override;

    void get_iter(int &iter) override { iter_ = iter; }

    void allocate_solver_shared_memory() override { allocate_shared_memory_P(); }

    /**
     * Determine whether anything will be executed in single precision
     * @param run_sp - [INPUT] Whether to switch to single precision
     */
    void single_prec_run(bool run_sp) override {
      if (run_sp) {
        if (!world_rank_) std::cerr << "Only single precision for the entire cuda GW solver is implemented. "
                                       "Input arguments \"P_sp\" and \"Sigma_sp\" will be ignored." << std::endl;
        sp_ = true;
      } else {
        sp_ = false;
      }
    }

    std::string name() const override  { return "cuda GW solver"; }

    void compute_Pq_from_P0q() override;

    const tensor_view<dcomplex, 1> &P0_q_t_QP() const override { return P0_q_t_QP_host_; }

  private:
    const transformer_t &ft_;

    const size_t max_kao_size_;
    const size_t max_qaux_size_;
    const size_t max_V_size_;

    const size_t nts_;
    const size_t nt_batch_;
    const size_t nw_b_;

    const std::string path_;

    bool sp_;
    int nqkpt_;
    double flop_count_;
    size_t iter_;

    // shared memory within a node, allocated in sc loop
    tensor_view<dcomplex, 3> &G_ts_kij_host_;
    tensor_view<dcomplex, 3> &Sigma_ts_kij_host_;

    // shared memory of P, allocated when initialize
    tensor_view<dcomplex, 1> P0_q_t_QP_host_;
    dcomplex *P0_data_;
    MPI_Win win_P_;

    execution_statistic_t gw_statistics;

    void GW_complexity_estimation();

    void allocate_shared_memory_P();

    void deallocate_shared_memory_P() {
      MPI_Win_free(&win_P_);
    }

    void GW_check_devices_free_space();

    void setup_solve();

    void gw_innerloop();

    template<typename prec>
    void compute_gw_selfenergy();

    // only compute P0, for testing purpose only
    template<typename prec>
    void compute_gw_P0();

    // compute sigma using P0, for testing purpose only
    template<typename prec>
    void compute_gw_selfenergy_P0();

    template<typename prec>
    void compute_P0q(std::unique_ptr<cugw_routine_block<prec> > &cugw_ptr,
                     mom_cons_callback &mom_cons, integral_reader_callback<prec> &reader);

    template<typename prec>
    void compute_Sigmak(std::unique_ptr<cugw_routine_block<prec> > &cugw_ptr,
                        mom_cons_callback &mom_cons, integral_reader_callback<prec> &reader);

    void obtain_full_P0q();

    void reduce_Pq();

    void init_events() {
      gw_statistics.add("Initialization");
      gw_statistics.add("GW_loop");
      gw_statistics.add("read");
      gw_statistics.add("P_reduce");
      gw_statistics.add("selfenergy_reduce");
      gw_statistics.add("total");

      gw_statistics.add("Compute P0");
      gw_statistics.add("Compute Sigma");
      gw_statistics.add("Compute P");
      gw_statistics.add("Copy data");
      gw_statistics.add("P0_contraction");
      gw_statistics.add("Sigma_contraction");
    }
  };

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_CUGW_SOLVER_BLOCK_H
