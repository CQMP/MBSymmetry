
#include "utils.h"
#include "dirac_character.h"

namespace symmetry_adaptation {

void DiracCharacter::calculate_conjugacy_class_sum(const std::vector<std::vector<int> > &conjugacy_classes,
                                                   const Ndarray<dcomplex> &rep) {
  // Temporary check. Will be generalized later.
  int dim = rep.dim();
  if (dim != 3) {
    throw std::runtime_error("calculate_conjugacy_class_sum only takes in rank 3 representation");
  }
  int n = rep.shape()[dim-1];
  int n_cc = conjugacy_classes.size();
  conjugacy_class_sum_.reserve(n_cc);
  relevant_class_idx_.resize(n_cc);
  Matrix<dcomplex> Omega_c(n, n);

  for (int i = 0; i < n_cc; ++i) {
    relevant_class_idx_[i] = i;
    Omega_c.setZero();
    for (int j = 0; j < conjugacy_classes[i].size(); ++j) {
      int x = conjugacy_classes[i][j];
      Omega_c += Ndarray_MatView(rep, n, n, x*n*n);
    }
    //std::cout << "conjugacy class: " << i << " sum: " << std::endl;
    //std::cout << Omega_c << std::endl;
    conjugacy_class_sum_.emplace_back(Omega_c);
  }
}

void DiracCharacter::calculate_proj_conjugacy_class_sum(const SpaceGroup &group,
                                                        const std::vector<int> &little_cogroup,
                                                        const std::vector<std::vector<int> > &conjugacy_classes,
                                                        const Ndarray<dcomplex> &proj_rep,
                                                        const Ndarray<dcomplex> &factor,
                                                        double tol) {
  // Temporary check. Will be generalized later.
  int dim = proj_rep.dim();
  if (dim != 3) {
    throw std::runtime_error("calculate_conjugacy_class_sum only takes in rank 3 representation");
  }
  int n = proj_rep.shape()[dim-1];
  int n_cc = conjugacy_classes.size();
  Matrix<dcomplex> Omega_c(n, n);

  for (int i = 0; i < n_cc; ++i) {
    const auto &conjugacy_class = conjugacy_classes[i];
    if (!check_relevance(group, little_cogroup, conjugacy_class, factor, tol)) {
      continue;
    }
    relevant_class_idx_.push_back(i);
    Omega_c.setZero();
    int c = conjugacy_class[0];  // gamma is an arbitrary element in the conjugacy class
    for (int a : conjugacy_class) {
      int b = -1;
      for (int x : little_cogroup) {
        if (group.multiplication_table()(a, x) == group.multiplication_table()(x, c)) {
          b = x;
          break;
        }
      } // b
      if (b == -1) {
        std::cout << std::endl;
        std::cout << a << " " << c << std::endl;
        std::cout << group.multiplication_table().row(a) << std::endl;
        std::cout << group.multiplication_table().col(c).transpose() << std::endl;
        throw std::runtime_error("Cannot find proper group operation when computing conjugacy sum of factor group");
      }
      Omega_c += factor.at(b, c) * std::conj(factor.at(a, b)) * Ndarray_MatView(proj_rep, n, n, a*n*n);
    } // a
    if (Omega_c.norm() < tol) {
      throw std::runtime_error("class sum is zero");
    }
    else {
      conjugacy_class_sum_.push_back(Omega_c);
    }
  }
  number_ = relevant_class_idx_.size();
  E_.resize(number_, size_);
}

bool DiracCharacter::check_relevance(const SpaceGroup &group,
                                     const std::vector<int> &little_cogroup,
                                     const std::vector<int> &conjugacy_class,
                                     const Ndarray<dcomplex> &factor,
                                     double tol) {

  for (int a : conjugacy_class) {
    for (int b : little_cogroup) {
      if (group.multiplication_table()(a, b) == group.multiplication_table()(b, a)) {
        if (std::abs(factor.at(a, b) - factor.at(b, a)) > tol) {
          return false;
        }
      }
    }
  }
  return true;
}

void DiracCharacter::compute_diagonal_elements() {
  for (int i = 0; i < conjugacy_class_sum_.size(); ++i) {
    E_.row(i) = (U_.adjoint() * conjugacy_class_sum_[i] * U_).diagonal();
  }
}

void DiracCharacter::find_block_size(double tol) {

  std::set<int> block_start_idx;
  block_start_idx.insert(0);

  for (int i = 0; i < number_; ++i) {
    for (int j = 1; j < size_; ++j) {
      if (std::abs(E_(i, j) - E_(i, j-1)) > tol){
        block_start_idx.insert(j);
      }
    }
  }
  block_start_idx_.assign(block_start_idx.begin(), block_start_idx.end());

  int n_blocks = block_start_idx.size();
  block_size_.resize(n_blocks);
  std::adjacent_difference(block_start_idx_.begin()+1, block_start_idx_.end(), block_size_.begin());
  block_size_[n_blocks-1] = size_ - block_start_idx_[n_blocks-1];
}

void DiracCharacter::save(alps::hdf5::archive &ar, const std::string &group) const {

  std::string prefix = group + "/Dirac_character/";

  ar[prefix + "U"] << U_;
  ar[prefix + "E"] << E_;
  ar[prefix + "block_start_idx"] << block_start_idx();
  ar[prefix + "block_size"] << block_size();
}

} // namespace symmetry_adaptation

