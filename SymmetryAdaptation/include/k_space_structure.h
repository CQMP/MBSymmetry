#ifndef SYMMETRYADAPTATION_STAR_H
#define SYMMETRYADAPTATION_STAR_H

#include "k_mesh.h"
#include "space_group.h"

namespace symmetry_adaptation {

class KSpaceStructure {

public:
  KSpaceStructure(const KMesh &k_mesh, const GroupBase &group, const TranslationVector &trans_vec, double tol=1e-9,
                  bool verbose=true, bool diag=true, bool time_reversal=true): k_mesh_(k_mesh), group_(group),
                  trans_vec_(trans_vec), nk_(k_mesh.size()), tol_(tol), time_reversal_(time_reversal) {
    generate_transform_table(verbose);
    find_stars();
    find_iBZ(verbose);
    find_ir_list();
    find_conjugate_relation();

    find_little_cogroup(verbose);
    if (diag) {
      find_little_cogroup_conjugacy_classes(verbose);
      find_little_cogroup_conjugacy_classes_mult(verbose);
    }
  };

  inline bool time_reversal() const { return time_reversal_; }

  inline int n_star() const { return stars_.size(); }

  inline const KMesh &k_mesh() const { return k_mesh_; }

  inline const TranslationVector &trans_vec() const { return trans_vec_; }

  inline const Matrix<int> &transform_table() const { return transform_table_; }

  inline const Matrix<int> &transform_table_rev() const { return transform_table_rev_; }

  inline const std::vector<std::vector<ColVector<double, DIM> > > &operator()() const { return stars_; }

  inline const std::vector<ColVector<double, DIM>> &operator()(int i) const { return stars_[i]; }

  inline int star_rep_index(int i) const { return stars_rep_index_[i]; }

  inline const std::vector<int> &star_rep_index() const { return stars_rep_index_; }

  inline const std::vector<int> &conj_check() const { return conj_check_;}

  // Return little cogroup of the ith star representative
  inline const std::vector<std::vector<int> > &star_little_cogroup_conjugacy_classes(int i) const {
    return star_little_cogroup_conjugacy_classes_[i];
  }

  // Return little cogroup of all the k point
  inline const std::vector<std::vector<int> > &little_cogroup() const {
    return little_cogroup_;
  }
  inline const std::vector<int > &little_cogroup(int i) const {
    return little_cogroup_[i];
  }

  // Return little cogroup of the ith k point
  inline const std::vector<std::vector<int> > &little_cogroup_conjugacy_classes(int i) const {
    return little_cogroup_conjugacy_classes_[i];
  }

  // Return little cogroup of the ith k point
  inline const std::vector<std::vector<int> > &little_cogroup_conjugacy_classes_mult(int i) const {
    return little_cogroup_conjugacy_classes_mult_[i];
  }

  void save(alps::hdf5::archive &ar, const std::string &group = "") const;

private:

  void generate_transform_table(bool verbose=true);
  void find_stars();
  void find_iBZ(bool verbose=true);
  std::vector< std::vector<int> > find_symm_op(std::vector<ColVector<double, DIM> > &star);
  void find_ir_list();
  void find_conjugate_relation();

  void find_little_cogroup(bool verbose=true);
  void find_little_cogroup_conjugacy_classes(bool verbose=true);
  void find_little_cogroup_conjugacy_classes_mult(bool verbose=true);

  const KMesh &k_mesh_;
  const GroupBase &group_;
  const TranslationVector &trans_vec_;

  const int nk_;
  const double tol_;
  const bool time_reversal_;

  // Dimension: ith star, jth point in star
  std::vector<std::vector<ColVector<double, DIM> > > stars_;
  // Number of k points in each star
  std::vector<int> weights_;
  // Dimension: ith star, jth point in star, group operation
  std::vector<std::vector<std::vector<int> > > stars_ops_;
  // Representative of each star
  std::vector<ColVector<double, DIM> > stars_rep_;
  // Index of representative in k mesh list
  std::vector<int> stars_rep_index_;
  // Index of k points in each star
  std::vector<std::vector<int> > stars_index_;
  // Operations that transform k points in iBZ to k points in full BZ
  std::vector<std::vector<int> > ks_ops_;
  // Corresponding k point in iBZ for all k points in full BZ, range from [0, nk)
  std::vector<int> ir_list_;
  // Corresponding index of k point in iBZ for all k points in full BZ, range from [0, n_star)
  std::vector<int> irre_index_;

  // -- Time reversal related
  // ith star is conjugate with jth star
  std::vector<int> stars_conj_index_;
  // List of star rep index after considering conjugation. FINAL STORED KPTS
  std::vector<int> irre_conj_list_;
  // Weight of each star after considering conjugation.
  std::vector<int> irre_conj_weight_;
  // ith k point is conjugate with jth k point
  std::vector<int> conj_index_;
  // Unique k points after considering conjugation
  std::vector<int> conj_list_;
  // Whether a k point is conjugated
  std::vector<int> conj_check_;
  // ith k point is in jth star after considering conjugation
  std::vector<int> irre_conj_index_;

  // Little cogroups of star representatives. Dimension: ith star, symmetry operation
  std::vector<std::vector<int> > star_little_cogroup_;
  // Little cogroups of all k points. Dimension: ith k point, symmetry operation
  std::vector<std::vector<int> > little_cogroup_;

  // Conjugacy classes of star little cogroups. Dimension: ith star, jth conjugacy class, symmetry operation
  std::vector<std::vector<std::vector<int> > > star_little_cogroup_conjugacy_classes_;
  // Conjugacy classes of k point little cogroups. Dimension: ith k point, jth conjugacy class, symmetry operation
  std::vector<std::vector<std::vector<int> > > little_cogroup_conjugacy_classes_;
  // Conjugacy classes of k point little cogroups computed from multiplication table.
  std::vector<std::vector<std::vector<int> > > little_cogroup_conjugacy_classes_mult_;

  // A(i, g) = j means element g transforms i to j
  Matrix<int> transform_table_;
  // A(j, g) = i means element g transforms i to j
  Matrix<int> transform_table_rev_;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_STAR_H
