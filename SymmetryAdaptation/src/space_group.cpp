
#include "space_group.h"
#include "spglib_anlys.h"

namespace symmetry_adaptation {

void SpaceGroup::get_space_group_info(const UnitCell &cell, bool diag) {

  SpglibDataset* dataset = spglib_anlys::get_symmetry_info(cell);

  h_ = dataset->n_operations;
  n_ = dataset->spacegroup_number;

  space_rep_.resize(h_);
  reciprocal_space_rep_.resize(h_);
  translations_.resize(h_);

  const Matrix<double, DIM, DIM> &A = cell.trans_vec().a_vectors();
  const Matrix<double, DIM, DIM> &A_inv = cell.trans_vec().a_vectors_inv();

  // temp matrix for storing rotation matrix with lattice vector as basis
  Matrix<double, DIM, DIM> rotation;
  Matrix<double, DIM, DIM> inv = -Matrix<double, DIM, DIM>::Identity();
  for (int g = 0; g < h_; ++g) {
    rotation = MatrixMap<int, DIM, DIM>((&(dataset->rotations[g][0][0]))).cast<double>();
    space_rep_[g] = A * rotation * A_inv;
    reciprocal_space_rep_[g] = space_rep_[g].inverse().transpose();
    translations_[g] = A * ColVectorMap<double, DIM>(&(dataset->translations[g][0]), DIM);
    if (translations_[g].cwiseAbs().maxCoeff() > tol_)
      symmorphic_ = false;

    if((rotation - inv).norm() < tol_)
        has_inversion_ = true;
  }

  int n_atoms = dataset->n_atoms;
  equiv_atoms_.resize(n_atoms);
  std::copy(dataset->equivalent_atoms, dataset->equivalent_atoms+n_atoms, equiv_atoms_.begin());

  spg_free_dataset(dataset);

  find_operation_inverse(&cell);

  if (diag)
    find_conj_classes();
}

void SpaceGroup::save(alps::hdf5::archive &ar, const std::string &group) const {
  std::string prefix = group + "/SpaceGroup/";
  ar[prefix + "n_operations"] << h_;
  ar[prefix + "spacegroup_number"] << n_;
  ar[prefix + "equiv_atoms"] << equiv_atoms_;
  ar[prefix + "symmorphic"] << symmorphic_;
  ar[prefix + "has_inversion"] << has_inversion_;
  // TODO: temporary solution. Will remove this after change all data structure to Ndarray
  Ndarray<double> translations(h_, DIM);
  Ndarray<double> space_rep(h_, DIM, DIM);
  Ndarray<double> reciprocal_space_rep(h_, DIM, DIM);
  for (int g = 0; g < h_; ++g) {
    Ndarray_VecView(translations, DIM, g * DIM) = translations_[g];
    Ndarray_MatView(space_rep, DIM, DIM, g * DIM * DIM) = space_rep_[g];
    Ndarray_MatView(reciprocal_space_rep, DIM, DIM, g * DIM * DIM) = reciprocal_space_rep_[g];
  }
  ar[prefix + "translations"] << translations;
  ar[prefix + "space_rep"] << space_rep;
  ar[prefix + "reciprocal_space_rep"] << reciprocal_space_rep;
}

void SpaceGroup::find_operation_inverse(const UnitCell *cell) {

  multiplication_table_.resize(h_, h_);
  op_inv_.resize(h_);

  Matrix<double, DIM, DIM> I = Matrix<double, DIM, DIM>::Identity();
  Matrix<double, DIM, DIM> mult;
  ColVector<double, DIM> translation;
  for (int g1 = 0; g1 < h_; ++g1) {
    for (int g2 = 0; g2 < h_; ++g2) {
      // multiply two operations: g2 * g1
      mult = space_rep_[g2] * space_rep_[g1];
      translation = space_rep_[g2] * translations_[g1] + translations_[g2];
      translation = cell->trans_vec().shift_back_to_center_cell(translation);
      // Find operation inverse
      if ((mult - I).norm() < tol_ && translation.norm() < tol_) {
        op_inv_[g1] = g2;
      }
      // Find element correspond to multiplied representation
      for (int g3 = 0; g3 < h_; ++g3) {
        if ((mult - space_rep_[g3]).norm() < tol_ && (translation - translations_[g3]).norm() < tol_) {
          multiplication_table_(g2, g1) = g3;
          break;
        }
        if (g3 == h_-1) {
          std::cout << mult << std::endl;
          std::cout << translation.transpose() << std::endl;
          throw std::runtime_error("could not find element corresponds to element "
          + std::to_string(g1) + " multiply " + std::to_string(g2));
        }
      }
    }
  }
}

void SpaceGroup::find_conj_classes() {

  std::vector<bool> conj_el_found(h_, false);
  int n = 0;
  for (int g = 0; g < h_; ++g) {
    if (conj_el_found[g]) continue;
    std::vector<int> new_conj_class;
    for (int a = 0; a < h_; ++a) {
      int againv = multiplication_table_(multiplication_table_(a, g), op_inv_[a]);
      if (!conj_el_found[againv]) {
        conj_el_found[againv] = true;
        new_conj_class.push_back(againv);
      }
    }
    conjugacy_classes_.push_back(new_conj_class);
    int size = new_conj_class.size();
    conjugacy_class_sizes_.push_back(size);
    n += size;
  }
  if (n != h_) {
    throw std::runtime_error("can not find conjugacy classes for all group elements");
  }
}

} // namespace symmetry_adaptation
