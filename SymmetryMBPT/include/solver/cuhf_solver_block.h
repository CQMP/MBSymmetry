#ifndef SYMMETRYMBPT_CUHF_SOLVER_BLOCK_H
#define SYMMETRYMBPT_CUHF_SOLVER_BLOCK_H

#include <iostream>
#include <iomanip>

#include "common.h"
#include "timing.h"
#include "mpi_utils.h"
#include "df_integral_block_t.h"
#include "solver_block_t.h"
#include "cuda_check.h"

namespace symmetry_mbpt {

  class cuhf_solver_block_t : public solver_block_t {
  public:
    cuhf_solver_block_t(MPI_Comm comm,
                        const params_t &p,
                        const symmetry_utils_t &symm_utils):
                        cuhf_solver_block_t(comm, symm_utils, p.ns, p.dfintegral_hf_file) {}
    cuhf_solver_block_t(MPI_Comm comm, const symmetry_utils_t &symm_utils,
                        int ns, const std::string &dfintegral_hf_file,
                        bool verbose = true, bool init_events = true) :
                        solver_block_t(comm, symm_utils, ns, dfintegral_hf_file) {
      check_for_cuda_mpi(world_comm_, world_rank_, devCount_per_node_);
      if (init_events) init_hf_events();
    }

    ~cuhf_solver_block_t() override {
      if(coul_int_reading_type_ == as_a_whole) {
        MPI_Win_free(&shared_win_);
      }
    }

    void solve() override {};

    tensor<dcomplex, 2> solve_HF(const tensor<dcomplex, 2>&, const tensor<dcomplex, 2>&,
                                 const tensor<dcomplex, 2>&, double madelung) override {};

    void get_iter(int &iter) override {};

    void set_shared_Coulomb() override {
      if (coul_int_reading_type_ == as_a_whole) {
        hf_statistics.start("Read");
        // Always read Coulomb integrals in double precision and cast them to single precision whenever needed
        read_entire_Coulomb_integrals(&Vk1k2_Qij_);
        hf_statistics.end("Read");
      } else {
        if (!world_rank_) std::cout << "Will read Coulomb integrals from chunks." << std::endl;
      }
      MPI_Barrier(world_comm_);
    };

    void allocate_solver_shared_memory() override {}

    /**
     * Determine whether anything will be executed in single precision
     * @param run_sp - [INPUT] Whether to switch to single precision
     */
    void single_prec_run(bool run_sp) override {
      std::cerr << "Single precision is not implemented for cuda HF." << std::endl;
    }

    std::string name() const override  { return "cuda HF block solver"; }

  protected:
    df_integral_block_t *coul_int_;

    MPI_Comm devices_comm_;
    int devices_rank_;
    int devices_size_;
    MPI_Win shared_win_;

    int devCount_total_;
    int devCount_per_node_;

    integral_reading_type coul_int_reading_type_ = as_a_whole;

    std::complex<double> *Vk1k2_Qij_;

    execution_statistic_t hf_statistics;

    /**
     * Setup inter-node, intra-node, and device communicators
     */
    void setup_MPI_structure();
    void clean_MPI_structure();

    /**
     * Update Coulomb integral stored in shared memory
     */
    void update_integrals(df_integral_block_t *coul_int, execution_statistic_t &statistics) {
      if (coul_int_reading_type_ == as_a_whole) {
        statistics.start("read whole integral");
        MPI_Win_fence(0, shared_win_);
        if (!intranode_rank_) coul_int->read_entire(Vk1k2_Qij_);  // TODO: check this
        MPI_Win_fence(0, shared_win_);
        statistics.end("read whole integral");
      }
    }

    /*
     * Read and allocate the entire Coulomb integral to MPI shared memory area. Collective behavior among _comm
     */
    template<typename prec>
    void read_entire_Coulomb_integrals(std::complex<prec> **Vk1k2_Qij) {
      long number_elements = symm_utils_.flat_V_size();
      MPI_Aint shared_buffer_size = number_elements * sizeof(std::complex<prec>);
      if (!world_rank_) {
        std::cout << std::setprecision(4);
        std::cout << "Reading the entire Coulomb integrals at once. Estimated memory requirement per node = "
                  << (double) shared_buffer_size / 1024 / 1024 / 1024 << " GB." << std::endl;
        std::cout << std::setprecision(15);
      }
      // Collective operations among _intranode_comm
      setup_mpi_shared_memory(Vk1k2_Qij, shared_buffer_size, shared_win_, intranode_comm_, intranode_rank_);
    }

  private:

    void init_hf_events() {
      hf_statistics.add("Read");
    }

  };

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_CUHF_SOLVER_BLOCK_H
