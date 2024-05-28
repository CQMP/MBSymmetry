#ifndef SYMMETRYADAPTATION_SPACE_GROUP_REP_H
#define SYMMETRYADAPTATION_SPACE_GROUP_REP_H

#include "type.h"
#include "space_group.h"
#include "unit_cell.h"
#include "k_space_structure.h"
#include "wigner_d.h"

namespace symmetry_adaptation {

namespace space_group_representation {

std::tuple<Matrix<int>, Ndarray<double> >
generate_permutation_matrix(const SpaceGroup &group, const UnitCell &unit_cell, double tol=1e-5, bool verbose=true);

Ndarray<dcomplex> generate_representation(const KSpaceStructure &k_struct, const SpaceGroup &group,
                                          const UnitCell &unit_cell, const WignerD &d_mat,
                                          double tol=1e-5, bool verbose=true);

} // namespace space_group_representation

class FactorGroupRep {

public:
  FactorGroupRep(const KSpaceStructure &k_struct,
                 const SpaceGroup &group,
                 const Ndarray<dcomplex> &representation,
                 double tol=1e-6) : k_struct_(k_struct), group_(group),
                                    representation_(representation), tol_(tol),
                                    factors_(k_struct_.k_mesh().size(), group_.order(), group_.order()),
                                    proj_rep_(representation.shape()) {
    compute_factor_system();
    generate_projective_representation();
  }
  ~FactorGroupRep() = default;

  inline const Ndarray<dcomplex> &proj_rep() const { return proj_rep_; }

  inline const Ndarray<dcomplex> &factors() const { return factors_; }

  void check_multiplication() const {
    check_factors(tol_);
    check_proj_representation(tol_);
  }

  std::vector<std::vector<std::vector<Matrix<dcomplex> > > > check_conjugate_relation() const;

private:
  const KSpaceStructure &k_struct_;
  const SpaceGroup &group_;
  const Ndarray<dcomplex> &representation_;
  double tol_;

  Ndarray<dcomplex> proj_rep_;
  Ndarray<dcomplex> factors_;

  void compute_factor_system();
  void generate_projective_representation();
  void check_factors(double tol) const;
  void check_proj_representation(double tol) const;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_SPACE_GROUP_REP_H
