
/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */


#ifndef SYMMETRYMBPT_SOLVER_FACTORY_T_H
#define SYMMETRYMBPT_SOLVER_FACTORY_T_H

#include <memory>
#include "solver_block_t.h"
#ifdef WITH_CUDA
#include "cuhf_solver_block.h"
#include "cugw_solver_block.h"
#endif

namespace symmetry_mbpt {

  class solver_factory_t {

  public:
    /**
     * Construct the many-body solver
     *
     * @param comm        - MPI communicator
     * @param p           - simulation parameters
     * @param _bz_utils   - Brillouin zone utils
     * @param _ft         - Transformer
     * @param _G_tau      - Green's function
     * @param _Selfenergy - Selfenergy
     * @return appropriate solver
     */
    static std::unique_ptr<solver_block_t> construct_solver(MPI_Comm comm, const params_t &p,
                                                            const symmetry_utils_t & symm_utils,
                                                            const transformer_t & ft,
                                                            tensor_view<dcomplex, 3> &G_tau,
                                                            tensor_view<dcomplex, 3> &Selfenergy) {
      switch(p.scf_type){
        case cuHF:
#ifdef WITH_CUDA
          return std::unique_ptr<solver_block_t>(new cuhf_solver_block_t(comm, p, symm_utils));
#else
          throw std::logic_error("Cuda is not enable. Please turn on WITH_CUDA option during compilation.");
#endif
        case cuGW:
#ifdef WITH_CUDA
          return std::unique_ptr<solver_block_t>(new cugw_solver_block_t(comm, p, ft, symm_utils, G_tau, Selfenergy));
#else
          throw std::logic_error("Cuda is not enable. Please turn on WITH_CUDA option during compilation.");
#endif
        default:
          throw std::logic_error("Unknown type of self-consistent solver");
      }
    }
  };
} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_SOLVER_FACTORY_T_H
