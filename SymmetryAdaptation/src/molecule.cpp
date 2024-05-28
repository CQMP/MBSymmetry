
#include <iostream>

#include "molecule.h"

namespace symmetry_adaptation {

void Molecule::construct(const std::string &xyz_str, bool verbose) {
  std::stringstream sstr(xyz_str);
  std::string line;
  std::getline(sstr, line, '\n');

  int natom = std::atoi(line.c_str());
  if (verbose)
    std::cout << "number of atoms: " << std::atoi(line.c_str()) << std::endl;

  std::getline(sstr, line, '\n');
  if (verbose)
    std::cout << "comment line: " << line << std::endl;

  std::string element_name;
  double px, py, pz;
  atoms_.reserve(natom);
  for (int i = 0; i < natom; ++i) {
    sstr >> element_name >> px >> py >> pz;
    if (verbose)
      std::cout << "element name: " << element_name << " position: " << px << " " << py << " " << pz << std::endl;
    atoms_.emplace_back(Atom(element_name, ColVector<double, DIM>(px, py, pz)));
  }
}

void Molecule::check_basis_complete(bool verbose) {
  for (const auto &atom : atoms_) {
    const auto &element_name = atom.element();
    auto it = basis_.expanded_desc().find(atom.element());
    if (it == basis_.expanded_desc().end()) {
      throw std::runtime_error("could not find element: " + element_name + " in basis.");
    }
    else {
      if (verbose) {
        std::cout << "for element: " << element_name << " found desc: " << it->first << std::endl;
        for (const auto & i : it->second) {
          std::cout << i << std::endl;
        }
      }
    }
  }
}

void Molecule::compute_shell_start_indices(bool verbose) {
  index_atom_n_l_.resize(atoms_.size());
  int current_index = 0;
  for (int i = 0; i < atoms_.size(); ++i) {
    const std::vector<orbital_def> &o = basis_.orbitals(atoms_[i].element());
    for (const auto &nl : o) {
      int n = nl.n;
      int l = nl.l;
      if (index_atom_n_l_[i].size() < n + 1) index_atom_n_l_[i].resize(n + 1);
      if (index_atom_n_l_[i][n].size() < l + 1) index_atom_n_l_[i][n].resize(l + 1, -1);
      index_atom_n_l_[i][n][l] = current_index;
      current_index += (2 * l + 1);
    }
  }
  n_orbitals_ = current_index;
  if (verbose)
    std::cout << "assigned a total of: " << current_index << " orbitals." << std::endl;
}

} // namespace symmetry_adaptation
