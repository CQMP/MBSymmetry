#ifndef SYMMETRYADAPTATION_MOLECULE_H
#define SYMMETRYADAPTATION_MOLECULE_H

#include <string>

#include "atom.h"
#include "basis.h"
#include "type.h"

namespace symmetry_adaptation {

class Molecule {

public:

  Molecule() = default;

  Molecule(const std::string &xyz_str, bool verbose=true) {
    construct(xyz_str, verbose);
  };

  Molecule(const std::string &xyz_str, const Basis &basis, bool verbose=true): basis_(basis){
    construct(xyz_str, verbose);
    check_basis_complete(verbose);
    compute_shell_start_indices(verbose);
  };

  virtual ~Molecule() = default;

  inline Basis &basis() { return basis_; }
  inline const Basis &basis() const { return basis_; }

  inline int n_atom() const { return atoms_.size(); }
  inline int n_orbitals() const { return n_orbitals_; }

  inline const Atom &atom(int i) const { return atoms_[i]; }

  inline const std::vector<std::vector<std::vector<int> > > &index_atom_n_l() const { return index_atom_n_l_; }
  int index_atom_n_l(int i, int n, int l) const { return index_atom_n_l_[i][n][l]; }

  void construct(const std::string &xyz_str, bool verbose=true);
  void check_basis_complete(bool verbose=true);
  void compute_shell_start_indices(bool verbose=true);

protected:

  std::vector<Atom> atoms_;
  Basis basis_;
  int n_orbitals_;
  std::vector<std::vector<std::vector<int> > > index_atom_n_l_;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_MOLECULE_H
