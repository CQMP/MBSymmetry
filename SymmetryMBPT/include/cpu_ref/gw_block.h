#ifndef SYMMETRYMBPT_GW_BLOCK_H
#define SYMMETRYMBPT_GW_BLOCK_H

#include "type.h"
#include "symmetry_utils.h"
#include "df_integral_block_t.h"

namespace symmetry_mbpt {

class gw_block_solver {
public:

  gw_block_solver(const symmetry_utils_t &symm_utils, tensor_view<dcomplex, 3> &G_ts_kij,
                  tensor_view<dcomplex, 3> &Sigma_ts_kij, int nts, int nts_full, int ns, int nw_b,
                  std::string prefix=""):
    symm_utils_(symm_utils), G_ts_kij_(G_ts_kij), Sigma_ts_kij_(Sigma_ts_kij),
    nk_(symm_utils.nk()), ink_(symm_utils.ink()), nts_(nts), nts_full_(nts_full),
    ns_(ns), nw_b_(nw_b), prefix_(prefix) {};
  ~gw_block_solver() {};

  void compute_full_P0_t(const std::string &df_integral_file);

  void compute_Sigma_from_P_t(const std::string &df_integral_file, const tensor<dcomplex, 2> &P_t_qQP);

  const tensor<dcomplex, 2> &P0_t_qQP() const { return P0_t_qQP_; }

private:

  int nk_;
  int ink_;

  int nts_;
  int nts_full_;
  int ns_;
  int nw_b_;

  const symmetry_utils_t& symm_utils_;
  tensor_view<dcomplex, 3> &G_ts_kij_;
  tensor_view<dcomplex, 3> &Sigma_ts_kij_;

  tensor<dcomplex, 2> P0_t_qQP_;

  tensor_view<dcomplex, 1> get_G(tensor_view<dcomplex, 3> &G, int t, int s, int k);
  tensor_view<dcomplex, 1> get_P(tensor<dcomplex, 2> &P, int t, int q);
  Ctensor_view<dcomplex, 1> get_const_P(const tensor<dcomplex, 2> &P, int t, int q);

  df_integral_block_t *coul_int_;
  std::string prefix_;
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_GW_BLOCK_H
