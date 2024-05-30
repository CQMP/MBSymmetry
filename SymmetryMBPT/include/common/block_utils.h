#ifndef SYMMETRYMBPT_UTILS_H
#define SYMMETRYMBPT_UTILS_H

#include "type.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

void G_block_to_flat(tensor<dcomplex, 3> &G_flat,
                     const tensor<dcomplex, 5> &G_block,
                     const symmetry_utils_t &symm_utils);

void P_block_to_flat(tensor<dcomplex, 2> &P_flat,
                     const tensor<dcomplex, 4> &P_block,
                     const symmetry_utils_t &symm_utils);

void G_flat_to_block(tensor<dcomplex, 5> &G_block,
                     const tensor<dcomplex, 3> &G_flat,
                     const symmetry_utils_t &symm_utils, size_t nao);

void P_flat_to_block(tensor<dcomplex, 4> &P_block,
                     const tensor<dcomplex, 2> &P_flat,
                     const symmetry_utils_t &symm_utils, size_t naux);

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_UTILS_H
