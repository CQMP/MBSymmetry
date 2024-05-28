#ifndef SYMMETRYADAPTATION_BASIS_H
#define SYMMETRYADAPTATION_BASIS_H

#include <set>
#include "type.h"

namespace symmetry_adaptation {

struct orbital_def {
  int n;
  int l;

  orbital_def(int n, int l) : n(n), l(l) {}
};

inline int l_index(char l) {
  if (l == 's') return 0;
  if (l == 'p') return 1;
  if (l == 'd') return 2;
  if (l == 'f') return 3;
  if (l == 'g') return 4;
  throw std::runtime_error("Only works for spdfg orbitals.");
}

class Basis {
public:
  Basis(int max_n=10): max_n_(max_n) {};
  ~Basis() = default;

  void read(const std::string &path, bool verbose=true);

  // this function should only be used for testing
  void add_atom(const std::string &atom, const std::string &desc);
  // this function is only used for testing
  void set_atom_types();

  inline const std::vector<std::string> &expanded_desc(const std::string &atom) const { return expanded_desc_.at(atom); }

  inline const std::map<std::string, std::vector<std::string>> &expanded_desc() const { return expanded_desc_; }

  // this function should only be used for testing
  inline std::map<std::string, int> &atom_types() { return atom_types_; }
  inline const std::map<std::string, int> &atom_types() const { return atom_types_; }

  inline const std::vector<orbital_def> &orbitals(const std::string &atom) const { return orbitals_.at(atom); }

  inline int n_orbitals(const std::string &atom) const { return n_orbitals_.at(atom); }

  inline int n_orbitals() const {
    int n = 0;
    for (const auto& a: n_orbitals_) n += a.second;
    return n;
  }

  inline int size() const { return n_orbitals_.size(); }

private:
  void find_all_orbitals(const std::string &atom, const std::string &desc);

  void parse_shell_ln(const std::string &atom, const std::map<int, std::vector<int> > &ln_chart);

  std::map<std::string, std::string> desc_;
  std::map<std::string, std::vector<std::string>> expanded_desc_;
  std::map<std::string, std::vector<orbital_def>> orbitals_;
  std::map<std::string, int> n_orbitals_;

  std::map<std::string, int> atom_types_;

  const int max_n_;
  static constexpr int max_l_ = 5;

  static const std::map<int, std::vector<std::string> > l_orbitals_;
};

inline std::ostream &operator<<(std::ostream &os, const Basis &b) {
  os << "basis: " << std::endl;
  os << "# of atom species: " << b.size() << std::endl;
  for (const auto& atom: b.expanded_desc()) {
    os << "atom name: " << atom.first << " basis size: " << atom.second.size() << std::endl;
    for (const auto& orbital: atom.second) {
      os << orbital << std::endl;
    }
  }
  return os;
}

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_BASIS_H
