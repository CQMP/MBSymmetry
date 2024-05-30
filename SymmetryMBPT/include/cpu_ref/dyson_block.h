#ifndef SYMMETRYMBPT_DYSON_BLOCK_H
#define SYMMETRYMBPT_DYSON_BLOCK_H

#include "type.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

class dyson_block {
public:
  dyson_block(const symmetry_utils_t &symm_utils,
              int ns, int nts, int nw): symm_utils_(symm_utils), ns_(ns), nts_(nts), nw_(nw),
                                        H_k_(ns, symm_utils.flat_ao_size()),
                                        S_k_(ns, symm_utils.flat_ao_size()),
                                        F_k_(ns, symm_utils.flat_ao_size()){};
  ~dyson_block() {};

  void read_input(const std::string &filename);

  void solve(tensor_view<dcomplex, 3> &G_tskij, const tensor_view<dcomplex, 3> &Sigma_tskij, double mu,
             const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
             const column<dcomplex> &wsample);

  void compute_G(tensor_view<dcomplex, 3> &G_tskij, const tensor_view<dcomplex, 3> &Sigma_tskij,
                 const tensor<dcomplex, 2> &F_k, const tensor<dcomplex, 2> &H_k, const tensor<dcomplex, 2> &S_k,
                 double mu, const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
                 const column<dcomplex> &wsample) const;

private:
  int ns_;
  int nts_;
  int nw_;

  tensor<dcomplex, 2> H_k_;
  tensor<dcomplex, 2> S_k_;
  tensor<dcomplex, 2> F_k_;

  const symmetry_utils_t &symm_utils_;
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_DYSON_BLOCK_H
