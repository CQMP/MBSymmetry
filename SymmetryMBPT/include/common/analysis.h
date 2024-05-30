#ifndef SYMMETRYMBPT_ANALYSIS_H
#define SYMMETRYMBPT_ANALYSIS_H

#include "symmetry_utils.h"
#include "params_t.h"
#include "transformer_t.h"
#include "common.h"

namespace symmetry_mbpt {

  double HF_complexity_estimation_full(int world_rank, const symmetry_utils_t &symm_utils, int ns, int ink=0,
                                       matmul_type mat_type=dcplx);

  double HF_complexity_estimation_block(int world_rank, const symmetry_utils_t &symm_utils, int ns, int ink=0,
                                        matmul_type mat_type=dcplx);

  double GW_complexity_estimation_full(int world_rank, const symmetry_utils_t &symm_utils,
                                       int ns, int nts, int nw_b, int ink=0, matmul_type mat_type=dcplx);

  double GW_complexity_estimation_block(int world_rank, const symmetry_utils_t &symm_utils,
                                        int ns, int nts, int nw_b, int ink=0, matmul_type mat_type=dcplx);

  inline double HF_complexity_estimation(int world_rank, const symmetry_utils_t &symm_utils, int ns,
                                         bool block, bool rot=false) {
    matmul_type mat_type = symm_utils.nk() == 1 ? dreal : dcplx;
    if (block) {
      if (rot)
        return HF_complexity_estimation_block(world_rank, symm_utils, ns, symm_utils.ink(), mat_type);
      else {
        if (symm_utils.rotate()) throw std::runtime_error("mismatch between symm utils and flop counts option");
        return HF_complexity_estimation_block(world_rank, symm_utils, ns, symm_utils.nk(), mat_type);
      }
    } else {
      if (rot)
        return HF_complexity_estimation_full(world_rank, symm_utils, ns, symm_utils.ink(), mat_type);
      else
        return HF_complexity_estimation_full(world_rank, symm_utils, ns, symm_utils.nk(), mat_type);
    }
  }

  inline double GW_complexity_estimation(int world_rank, const symmetry_utils_t &symm_utils, const params_t &p,
                                         bool block, bool rot=false) {
    matmul_type mat_type = symm_utils.nk() == 1 ? dreal : dcplx;
    transformer_t transformer(p);
    if (block) {
      if (rot)
        return GW_complexity_estimation_block(world_rank, symm_utils, p.ns, p.ni+2, transformer.nw_b(),
                                              symm_utils.ink(), mat_type);
      else {
        if (symm_utils.rotate()) throw std::runtime_error("mismatch between symm utils and flop counts option");
        return GW_complexity_estimation_block(world_rank, symm_utils, p.ns, p.ni+2, transformer.nw_b(),
                                              symm_utils.nk(), mat_type);
      }
    } else {
      if (rot)
        return GW_complexity_estimation_full(world_rank, symm_utils, p.ns, p.ni+2, transformer.nw_b(),
                                             symm_utils.ink(), mat_type);
      else
        return GW_complexity_estimation_full(world_rank, symm_utils, p.ns, p.ni+2, transformer.nw_b(),
                                             symm_utils.nk(), mat_type);
    }
  }

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_ANALYSIS_H
