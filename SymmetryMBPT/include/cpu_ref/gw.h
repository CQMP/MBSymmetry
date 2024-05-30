#ifndef SYMMETRYMBPT_GW_H
#define SYMMETRYMBPT_GW_H

#include "type.h"
#include "df_integral_t.h"

namespace symmetry_mbpt {

class gw_solver {
public:

  gw_solver(const symmetry_utils_t &symm_utils, tensor_view<dcomplex, 5> &G_tskij,
            tensor_view<dcomplex, 5> &Sigma_tskij, int nao, int NQ, int nts, int nts_full, int ns, int nw_b,
            std::string prefix="", bool symm=true):
    symm_utils_(symm_utils), G_tskij_(G_tskij), Sigma_tskij_(Sigma_tskij),
    nao_(nao), NQ_(NQ), nk_(symm_utils.nk()), nts_(nts), nts_full_(nts_full),
    ns_(ns), nw_b_(nw_b), P0_tQP_(nts, 1, NQ, NQ), P0_wQP_(nw_b, 1, NQ, NQ), prefix_(prefix), symm_(symm) {};
  ~gw_solver() {};

  const tensor<dcomplex, 4> &P0_tqQP() const { return P0_tqQP_; }

  tensor<dcomplex, 4> &P0_tqQP() { return P0_tqQP_; }

  void solve(const std::string &df_integral_file, const MMatrixX<dcomplex> &T_wt, const MMatrixX<dcomplex> &T_tw);

  void compute_full_P0_t(const std::string &df_integral_file);

  void compute_Sigma_from_P_t(const std::string &df_integral_file, const tensor<dcomplex, 4> &P_tqQP);

  void compute_P_from_P0(const MatrixX<dcomplex> &T_wt, const MatrixX<dcomplex> &T_tw);

  template<typename T>
  static void P0_contraction(MMatrixX<T> &Pq0_QP, const MMatrixX<T> &G_tp, const MMatrixX<T> &G1_mn,
                             const tensor<T, 3> &V_Pqm,
                             tensor<T, 3> &X1_tmQ, tensor<T, 3> &X2_Ptm, tensor<T, 3> &Y_inP, int nao, int NQ);

  template<typename T>
  static void Sigma_contraction(MMatrixX<T> &Sigma_ij,
                                const MMatrixX<T> &G1_mn, const CMMatrixX<T> &P_QP, const tensor<T, 3> &V_Qim,
                                tensor<T, 3> &V_nPj, tensor<T, 3> &Y1_Qin, tensor<T, 3> &Y2_inP, int nao, int NQ);

  template<typename T>
  static void eval_P_from_P0(tensor<T, 4> &P_tQP,
                             const MMatrixX<T> &T_wt, const MMatrixX<T> &T_tw,
                             tensor<T, 4> &P0_wQP, int nw, int nt, int NQ);

  template<size_t N>
  void make_hermition(tensor<dcomplex, N> &X) {
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(X.shape().begin(), X.shape().end()-2, 1ul, std::multiplies<size_t>());
    size_t nao = X.shape()[N-1];
    for (size_t i = 0; i < dim1; ++i) {
      MMatrixXcd  Xm(X.data() + i * nao * nao, nao, nao);
      Xm = 0.5 * (Xm + Xm.conjugate().transpose().eval());
    }
  }

private:
  int nao_;
  int NQ_;

  int nk_;
  //int ink_;

  int nts_;
  int nts_full_;
  int ns_;
  int nw_b_;

  const symmetry_utils_t& symm_utils_;
  tensor_view<dcomplex, 5> &G_tskij_;
  tensor_view<dcomplex, 5> &Sigma_tskij_;

  tensor<dcomplex, 4> P0_tQP_;
  tensor<dcomplex, 4> P0_wQP_;

  tensor<dcomplex, 4> P0_tqQP_; // for testing purpose only

  df_integral_t *coul_int_;
  std::string prefix_;
  bool symm_;

  MMatrixX<dcomplex> get_G(tensor_view<dcomplex, 5> &G, int t, int s, int k) const;
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_GW_H
