
#include <iostream>
#include "spglib_anlys.h"

namespace symmetry_adaptation {

namespace spglib_anlys {

SpglibDataset* get_symmetry_info(const UnitCell &cell, double tol) {

  double lattice[DIM][DIM];
  MatrixMap<double, DIM, DIM>(&lattice[0][0], DIM, DIM) = cell.trans_vec().a_vectors();

  int n_atom = cell.n_atom();
  double position[n_atom][DIM];
  std::vector<int> types(n_atom);
  for (int i = 0; i < n_atom; ++i) {
    // Need to get fraction position
    ColVectorMap<double>(&position[0][0]+i*DIM, DIM) = cell.trans_vec().a_vectors_inv() * cell.atom(i).pos();
    types[i] = cell.basis().atom_types().find(cell.atom(i).element())->second;
  }

  SpglibDataset *dataset = spg_get_dataset(lattice, position, types.data(), n_atom, tol);

  return dataset;
}

} // namespace spglib_anlys

} // namespace symmetry_adaptation
