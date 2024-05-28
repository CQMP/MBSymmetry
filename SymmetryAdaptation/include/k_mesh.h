#ifndef SYMMETRYADAPTATION_K_MESH_H
#define SYMMETRYADAPTATION_K_MESH_H

#include <string>

#include "type.h"
#include "translation_vector.h"
#include "utils.h"

namespace symmetry_adaptation {

class KMesh {
public:

  KMesh(int q, const TranslationVector &trans_vec) : q_(q), trans_vec_(trans_vec),
                                                     self_conjugate_(std::pow(q_, DIM), false) {};
  ~KMesh() = default;

  inline const std::vector<ColVector<double, DIM> > &kpts() const { return kpts_; }

  inline const std::vector<ColVector<double, DIM> > &kpts_scaled() const { return kpts_scaled_; }

  inline const ColVector<double, DIM> &kpt(int i) const { return kpts_[i]; }

  inline const std::vector<bool> &self_conjugate() const { return self_conjugate_; }

  inline int size() const { return kpts_.size(); }

  void read_k_mesh(const std::string &h5file, double tol=1e-9);

  void generate_k_mesh(const std::string &convention = "pyscf", double tol=1e-9);

  static int find_k_index(const std::vector<ColVector<double, DIM> > &kmesh,
                          const ColVector<double> &kpt, double tol=1e-9);

  template<int D>
  static void wrap_kpt_to_BZ(ColVector<double, D> &k, const TranslationVector &trans_vec) {
    ColVector<double, D> k_scaled = trans_vec.b_vectors_inv() * k;
    wrap_kpt_scaled_to_BZ(k_scaled);
    k = trans_vec.b_vectors() * k_scaled;
  }

  template<int D>
  static void wrap_kpt_scaled_to_BZ(ColVector<double, D> &kpts, double tol= 1e-9) {
    std::for_each(kpts.data(), kpts.data()+kpts.size(),
                  [=](double &n){ if (n < -(0.5+tol)) n += 1; else if (n - (0.5-tol) > 0.0) n -= 1; });
  }

  void save(alps::hdf5::archive &ar, const std::string &group = "") const;

private:

  const int q_;
  const TranslationVector &trans_vec_;
  std::vector<ColVector<double, DIM> > kpts_;
  std::vector<ColVector<double, DIM> > kpts_scaled_;

  // Whether a k point is its own conjugate (minus sign)
  std::vector<bool> self_conjugate_;

  void find_self_conjugate_kpts(double tol=1e-9);
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_K_MESH_H
