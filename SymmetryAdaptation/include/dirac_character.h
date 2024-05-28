#ifndef SYMMETRYADAPTATION_DIRAC_CHARACTER_H
#define SYMMETRYADAPTATION_DIRAC_CHARACTER_H

#include <vector>
#include <alps/hdf5.hpp>

#include "type.h"
#include "space_group.h"

namespace symmetry_adaptation {

class DiracCharacter {

public:

  DiracCharacter(int number, int size) : number_(number), size_(size), U_(size, size), E_(number, size) {};
  ~DiracCharacter() = default;

  inline const std::vector<Matrix<dcomplex> > &conjugacy_class_sum() const { return conjugacy_class_sum_; }
  inline std::vector<Matrix<dcomplex> > &conjugacy_class_sum() { return conjugacy_class_sum_; }

  inline const Matrix<dcomplex> &U() const { return U_; }
  inline Matrix<dcomplex> &U() { return U_; }

  inline const Matrix<dcomplex> &E() const { return E_; }
  inline Matrix<dcomplex> &E() { return E_; }

  inline int size() const { return size_; }

  inline const std::vector<int> &block_size() const { return block_size_; }
  inline const std::vector<int> &block_start_idx() const { return block_start_idx_; }

  // Given a rep, calculate conjugacy class sum using conjugacy_classes_ info
  void calculate_conjugacy_class_sum(const std::vector<std::vector<int> > &conjugacy_classes,
                                     const Ndarray<dcomplex> &rep);

  void calculate_proj_conjugacy_class_sum(const SpaceGroup &group,
                                          const std::vector<int> &little_cogroup,
                                          const std::vector<std::vector<int> > &conjugacy_classes,
                                          const Ndarray<dcomplex> &proj_rep,
                                          const Ndarray<dcomplex> &factor, double tol=1e-9);
  void compute_diagonal_elements();

  void find_block_size(double tol = 1e-10);

  void save(alps::hdf5::archive &ar, const std::string &group = "") const;
  
private:

  // Number of conjugacy classes
  int number_;
  // Size of orep
  int size_;

  std::vector<Matrix<dcomplex> > conjugacy_class_sum_;
  std::vector<int> relevant_class_idx_;

  // Unitary matrix that simultaneous diagonalize the dirac characters
  Matrix<dcomplex> U_;
  // Diagonal part of diagonalized Matrix.
  Matrix<dcomplex> E_;
  // Block size for different irreps
  std::vector<int> block_size_;
  // Starting index for different blocks
  std::vector<int> block_start_idx_;

  // Check whether a class is relevant
  static bool check_relevance(const SpaceGroup &group,
                              const std::vector<int> &little_cogroup,
                              const std::vector<int> &conjugacy_class,
                              const Ndarray<dcomplex> &factor,
                              double tol=1e-9);
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_DIRAC_CHARACTER_H
