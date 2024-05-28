#ifndef SYMMETRYADAPTATION_GROUP_BASE_H
#define SYMMETRYADAPTATION_GROUP_BASE_H

#include "type.h"
#include "utils.h"
#include "unit_cell.h"

namespace symmetry_adaptation {

class GroupBase {

public:

  GroupBase(double tol=1e-12): tol_(tol), h_(0), n_(-1) {};
  virtual ~GroupBase() = default;

  // Get order of the group
  inline int order() const { return h_; }
  inline int h() const { return h_; }

  inline const Matrix<int> &multiplication_table() const { return multiplication_table_; }

  inline const std::vector<Matrix<double, DIM, DIM> > &space_rep() const { return space_rep_; }

  inline const Matrix<double, DIM, DIM> &space_rep(int g) const { return space_rep_[g]; }

  inline const std::vector<Matrix<double, DIM, DIM> > &reciprocal_space_rep() const { return reciprocal_space_rep_; }

  inline const Matrix<double, DIM, DIM> &reciprocal_space_rep(int g) const { return reciprocal_space_rep_[g]; }

  inline const std::vector<int> &conjugacy_class_sizes() const { return conjugacy_class_sizes_; }

  inline const std::vector<std::vector<int> > &conjugacy_classes() const { return conjugacy_classes_; }

  inline const std::vector<int> &operation_inverse() const { return op_inv_; }

  inline const int &operation_inverse(int g) const { return op_inv_[g]; }

protected:

  virtual void find_operation_inverse(const UnitCell *cell = nullptr) = 0;

  double tol_;

  // Order of the group
  int h_;
  // Number of the group
  int n_;

  Matrix<int> multiplication_table_;
  std::vector<Matrix<double, DIM, DIM> > space_rep_;
  std::vector<Matrix<double, DIM, DIM> > reciprocal_space_rep_;
  // Table of inverse operation
  std::vector<int> op_inv_;

  std::vector<int> conjugacy_class_sizes_;
  // Nested vector here since dimensions are different
  std::vector<std::vector<int> > conjugacy_classes_;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_GROUP_BASE_H
