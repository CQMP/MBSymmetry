#ifndef SYMMETRYADAPTATION_SPACE_GROUP_H
#define SYMMETRYADAPTATION_SPACE_GROUP_H

#include "type.h"
#include "utils.h"
#include "unit_cell.h"
#include "group_base.h"

namespace symmetry_adaptation {

class SpaceGroup: public GroupBase {

public:

  SpaceGroup(double tol=1e-12): GroupBase(tol), symmorphic_(true), has_inversion_(false) {};
  ~SpaceGroup() override = default;

  void get_space_group_info(const UnitCell &cell, bool diag=false);

  inline int n() const { return n_; }
  inline int space_group_number() const { return n_; }

  inline const std::vector<ColVector<double, DIM> > &translations() const { return translations_; }

  inline const ColVector<double, DIM> &translation(int g) const { return translations_[g]; }

  inline const std::vector<int> &equiv_atoms() const { return equiv_atoms_; }

  inline bool symmorphic() const { return symmorphic_; }
  inline bool has_inversion() const { return has_inversion_; }

  void save(alps::hdf5::archive &ar, const std::string &group = "") const;

private:

  void find_operation_inverse(const UnitCell *cell) override;

  void find_conj_classes();

  std::vector<ColVector<double, DIM> > translations_;
  std::vector<int> equiv_atoms_;
  bool symmorphic_;
  bool has_inversion_;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_SPACE_GROUP_H
