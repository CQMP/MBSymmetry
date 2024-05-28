
#include <iostream>
#include <fstream>

#include "basis.h"

namespace symmetry_adaptation {

const std::map<int, std::vector<std::string> > Basis::l_orbitals_{
    {0, {"s"}},
    {1, {"px",   "py",   "pz"}},
    {2, {"dxy",  "dyz",  "dz^2",  "dxz",  "dx2-y2"}},
    {3, {"fy^3", "fxyz", "fyz^2", "fz^3", "fxz^2", "fzx^2", "fx^3"}},
    {4, {"g-4", "g-3", "g-2", "g-1", "g0", "g1", "g2", "g3", "g4"}}
};

void Basis::read(const std::string &path, bool verbose) {

  if (verbose)
    std::cout << "reading basis information" << std::endl;
  std::ifstream bfile(path.c_str());
  if (!bfile.good()) throw std::runtime_error("basis file could not be opened.");

  std::string line;
  std::string atom, basis;
  int ntype = 0;
  while (std::getline(bfile, line)) {
    std::istringstream iss(line);
    iss >> atom >> basis;
    if (verbose)
      std::cout << "element: " << atom << " basis: " << basis << std::endl;
    add_atom(atom, basis);
    atom_types_[atom] = ntype;
    ntype += 1;
  }
  std::cout << std::endl;
};

void Basis::add_atom(const std::string& atom, const std::string& desc) {
  desc_[atom] = desc;
  find_all_orbitals(atom, desc);
}

void Basis::set_atom_types() {
  int ntype = 0;
  for (const auto& atom: desc_) {
    atom_types_[atom.first] = ntype;
    ntype += 1;
  }
}

void Basis::find_all_orbitals(const std::string &atom, const std::string &desc) {

  expanded_desc_[atom] = std::vector<std::string>();
  orbitals_[atom] = std::vector<orbital_def>();
  n_orbitals_[atom] = 0;

  std::vector<size_t> n_pos(max_n_);
  std::vector<size_t> exist_n;
  for (int i = 0; i < max_n_; ++i) {
    std::string n_number = std::to_string(i + 1);
    size_t pos = desc.find(n_number);
    n_pos[i] = pos;
    if (pos < desc.length())
      exist_n.emplace_back(i);
  }
  std::cout << std::endl;
  if (n_pos[max_n_ - 1] < desc.length()) {
    std::cout << "current max_n: " << max_n_ << std::endl;
    throw std::runtime_error("increase max_n in basis");
  }

  std::map<int, std::vector<int> > ln_chart;
  for (int l = 0; l < max_l_; ++l) {
    ln_chart.emplace(l, std::vector<int>(0));
  }

  for (const auto &n: exist_n) {
    std::size_t str_len = std::min(n_pos[n + 1], desc.length()) - n_pos[n];
    const std::string substr = desc.substr(n_pos[n], str_len);
    for (int i = std::to_string(n+1).length(); i < substr.length(); ++i) {
      ln_chart[l_index(substr[i])].emplace_back(n + 1);
    }
  }

  parse_shell_ln(atom, ln_chart);
}

void Basis::parse_shell_ln(const std::string &atom, const std::map<int, std::vector<int> > &ln_chart) {
  for (int l = 0; l < max_l_; ++l) {
    const auto &n_index = ln_chart.at(l);
    const auto &l_orb = l_orbitals_.at(l);

    for (const auto &ni: n_index) {
      std::string n = std::to_string(ni);
      for (const auto &lo: l_orb) {
        expanded_desc_[atom].emplace_back(atom + " " + n + lo);
      }
      orbitals_[atom].emplace_back(orbital_def(ni, l));
      n_orbitals_[atom] += 2 * l + 1;
    }
  }
}

} // namespace symmetry_adaptation