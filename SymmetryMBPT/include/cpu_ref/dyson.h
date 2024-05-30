#ifndef SYMMETRYMBPT_DYSON_H
#define SYMMETRYMBPT_DYSON_H

#include "type.h"

namespace symmetry_mbpt {

class dyson {

public:
  dyson(int ns, int nts, int nw, int nk, int nao): ns_(ns), nts_(nts), nw_(nw), nk_(nk), nao_(nao),
                                                   H_k_(ns, nk, nao, nao), S_k_(ns, nk, nao, nao),
                                                   F_k_(ns, nk, nao, nao) {};
  ~dyson() {};

  void read_input(const std::string &filename);

  void solve(tensor_view<dcomplex, 5> &G_tskij, const tensor_view<dcomplex, 5> &Sigma_tskij, double mu,
             const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
             const column<dcomplex> &wsample);

  void compute_G(tensor_view<dcomplex, 5> &G_tskij, const tensor_view<dcomplex, 5> &Sigma_tskij,
                 const tensor<dcomplex, 4> &F_k, const tensor<dcomplex, 4> &H_k, const tensor<dcomplex, 4> &S_k,
                 double mu, const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw,
                 const column<dcomplex> &wsample) const;

  const tensor<dcomplex, 4> &H_k() const { return H_k_; };
  const tensor<dcomplex, 4> &S_k() const { return S_k_; };

private:
  int ns_;
  int nts_;
  int nw_;
  int nk_;
  int nao_;

  tensor<dcomplex, 4> H_k_;
  tensor<dcomplex, 4> S_k_;
  tensor<dcomplex, 4> F_k_;
};

} // symmetry_mbpt

#endif //SYMMETRYMBPT_DYSON_H
