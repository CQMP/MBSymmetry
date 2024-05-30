#ifndef SYMMETRYMBPT_SYMMETRY_UTILS_H
#define SYMMETRYMBPT_SYMMETRY_UTILS_H

#include <algorithm>
#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include "type.h"
#include "params_t.h"

namespace symmetry_mbpt {

enum IntegralType {
  first_integral, second_direct, second_exchange
};

class symmetry_utils_t {

public:
  explicit symmetry_utils_t(const params_t &p): symmetry_utils_t(p.dfintegral_file, p.nk,
                                                                 p.symmetry_file, true,
                                                                 p.symmetry_rot_file,
                                                                 p.rotate, p.time_reversal) {}
  symmetry_utils_t(const std::string &dfintegral_file, size_t nk,
                   const std::string &symm_file_name = "",
                   bool read_kpair = false,
                   // rotation related
                   const std::string &rot_file_name = "",
                   bool rotate=false, bool time_reversal=false);

  size_t nk() const { return nk_; }

  size_t ink() const { return ink_; }

  double nkpw() const { return nkpw_; }

  size_t num_kpair_stored() const { return num_kpair_stored_; }

  const tensor<long, 2> &k_pairs() const { return k_pairs_; }

  const tensor<double, 2> &kmesh() const { return kmesh_; }

  bool rotate() const { return rotate_; }
  bool time_reversal() const { return time_reversal_; }

  // -- block diagonalization related
  const tensor<int, 1> &ao_sizes(int i) const { return ao_sizes_[i]; }
  const tensor<int, 1> &aux_sizes(int i) const { return aux_sizes_[i]; }

  const tensor<int, 1> &ao_offsets(int i) const { return ao_offsets_[i]; }
  const tensor<int, 1> &aux_offsets(int i) const { return aux_offsets_[i]; }

  const tensor<int, 1> &ao_block_offsets(int i) const { return ao_block_offsets_[i]; }
  const tensor<int, 1> &aux_block_offsets(int i) const { return aux_block_offsets_[i]; }

  const tensor<int, 1> &kao_slice_offsets() const { return kao_slice_offsets_; }
  const tensor<int, 1> &qaux_slice_offsets() const { return qaux_slice_offsets_; }

  const tensor<int, 1> &kao_slice_sizes() const { return kao_slice_sizes_; }
  const tensor<int, 1> &qaux_slice_sizes() const { return qaux_slice_sizes_; }

  const tensor<int, 1> &kao_slice_offsets_irre() const { return kao_slice_offsets_irre_; }
  const tensor<int, 1> &kao_slice_sizes_irre() const { return kao_slice_sizes_irre_; }
  const tensor<int, 1> &ao_sizes_irre(int i) const { return ao_sizes_irre_[i]; }
  const tensor<int, 1> &ao_offsets_irre(int i) const { return ao_offsets_irre_[i]; }
  const tensor<int, 1> &qaux_slice_offsets_irre() const { return qaux_slice_offsets_irre_; }
  const tensor<int, 1> &qaux_slice_sizes_irre() const { return qaux_slice_sizes_irre_; }
  const tensor<int, 1> &aux_sizes_irre(int i) const { return aux_sizes_irre_[i]; }
  const tensor<int, 1> &aux_offsets_irre(int i) const { return aux_offsets_irre_[i]; }

  const tensor<int, 1> &korep_slice_offsets() const { return korep_slice_offsets_; }
  const tensor<int, 1> &korep_slice_sizes() const { return korep_slice_sizes_; }
  const tensor<int, 1> &kauxrep_slice_offsets() const { return kauxrep_slice_offsets_; }
  const tensor<int, 1> &kauxrep_slice_sizes() const { return kauxrep_slice_sizes_; }

  size_t max_kao_size() const { return max_kao_size_; }
  size_t max_qaux_size() const { return max_qaux_size_; }
  size_t max_kpair_size() const { return max_kpair_size_; }

  const tensor<int, 1> &V_offsets(int k1, int k2) const {
    size_t idx = k1 * nk_ + k2;
    return tensor_offsets_[idx];
  }

  const tensor<int, 2> &V_irreps(int k1, int k2) const {
    size_t idx = k1 * nk_ + k2;
    return tensor_irreps_[idx];
  }

  /**
   * Find position in irre_list
   */
  /*size_t irre_pos(size_t k) const {
    auto itr = std::find(irre_list_.begin(), irre_list_.end(), k);
    size_t index = std::distance(irre_list_.begin(), itr);
    return index;
  }*/

  size_t irre_pos_kpair(size_t idx) const {
    return idx;
  }

  const std::vector<long> &irre_list() const { return irre_conj_list_; }
  const std::vector<double> &weight() const { return irre_conj_weight_; }
  const std::vector<long> &irre_index() const { return irre_conj_index_; }
  const std::vector<long> &conj_index() const { return conj_index_; }
  const std::vector<long> &index() const { return index_; }
  const std::vector<std::vector<long> > &kpts_ops() const { return kpts_ops_; }

  std::array<size_t, 4> momentum_conservation(const std::array<size_t, 3> &in,
                                              IntegralType type = first_integral) const;

  // block interaction tensor related
  long kpair_slice_offsets(size_t idx) const { return kpair_slice_offsets_(idx); }
  long kpair_slice_sizes(size_t idx) const { return kpair_slice_sizes_(idx); }
  long kpair_slice_offsets(int k1, int k2) const {
    size_t idx = k1 * nk_ + k2;
    return kpair_slice_offsets(idx);
  }
  long kpair_slice_sizes(int k1, int k2) const {
    size_t idx = k1 * nk_ + k2;
    return kpair_slice_sizes(idx);
  }

  // full size of flattened matrix in ao space (k points in iBZ)
  size_t flat_ao_size() const { return kao_slice_offsets_irre_(ink_-1)+kao_slice_sizes_irre_(ink_-1); }
  // full size of flattened matrix in aux space (q points in iBZ)
  size_t flat_aux_size() const { return qaux_slice_offsets_irre_(ink_-1)+qaux_slice_sizes_irre_(ink_-1); }
  // full size of flattened interaction tensor
  long flat_V_size() const {
    return kpair_slice_offsets_(num_kpair_stored_-1)+kpair_slice_sizes(num_kpair_stored_-1);
  }
  // full size of flattened matrix in ao space (all k points)
  size_t flat_ao_size_full() const { return kao_slice_offsets_(nk_-1)+kao_slice_sizes_(nk_-1); }

  size_t flat_orep_size() const { return korep_slice_offsets_(nk_-1)+korep_slice_sizes_(nk_-1); }
  size_t flat_auxrep_size() const { return kauxrep_slice_offsets_(nk_-1)+kauxrep_slice_sizes_(nk_-1); }

  // sizes of all irreps of any k point should add up to nao
  int nao() const { return ao_sizes_[0].vector().sum(); }
  int NQ() const { return aux_sizes_[0].vector().sum(); }

  // rotation matrices
  const tensor<dcomplex, 1> &kspace_orep() const { return kspace_orep_; }
  const tensor<dcomplex, 1> &kspace_auxrep() const { return kspace_auxrep_trans_; }

  const std::vector<tensor<int, 2> > &orep_irreps() const { return orep_irreps_; }
  const std::vector<tensor<int, 1> > &orep_offsets() const { return orep_offsets_; }
  const std::vector<tensor<int, 2> > &orep_sizes() const { return orep_sizes_; }
  const std::vector<tensor<int, 2> > &auxrep_irreps() const { return auxrep_irreps_; }
  const std::vector<tensor<int, 1> > &auxrep_offsets() const { return auxrep_offsets_; }
  const std::vector<tensor<int, 2> > &auxrep_sizes() const { return auxrep_sizes_; }

private:

  size_t nk_;
  size_t ink_;
  size_t num_kpair_stored_;

  double nkpw_;

  tensor<int, 2> q_ind_;
  tensor<int, 2> q_ind2_;

  tensor<double, 2> kmesh_;

  bool rotate_;
  bool time_reversal_;

  // -- rotation related
  // irreducible k-point list
  std::vector<long> irre_list_; // ink
  // corresponding weight
  std::vector<double> weight_; // ink
  // corresponding irreducible index
  std::vector<long> index_; // nk
  // corresponding irreducible index in iBZ
  std::vector<long> irre_index_; // nk
  // -- time reversal related
  // irreducible k-point list after considering conjugation. FINAL STORED KPTS
  std::vector<long> irre_conj_list_; // ink
  // corresponding weight after considering conjugation
  std::vector<double> irre_conj_weight_; // ink
  // ith k point is conjugate with jth k point
  std::vector<long> conj_index_; // nk
  // corresponding irreducible index in iBZ after considering conjugation
  std::vector<long> irre_conj_index_; // nk

  tensor<long, 2> k_pairs_;

  size_t mom_cons(size_t i, size_t j, size_t k) const;

  // -- block diagonalization related
  // sizes of each irrep
  std::vector<tensor<int, 1> > ao_sizes_;  // (k, sizes)
  std::vector<tensor<int, 1> > aux_sizes_;  // (k, sizes)
  // offsets in flattened vector
  std::vector<tensor<int, 1> > ao_offsets_;  // (k, offsets)
  std::vector<tensor<int, 1> > aux_offsets_;  // (k, offsets)
  // offsets in block matrix, used for testing purpose only
  std::vector<tensor<int, 1> > ao_block_offsets_;  // (k, offsets)
  std::vector<tensor<int, 1> > aux_block_offsets_;  // (k, offsets)
  // start index of slices of different k points
  tensor<int, 1> kao_slice_offsets_;
  tensor<int, 1> qaux_slice_offsets_;
  tensor<int, 1> kao_slice_sizes_;
  tensor<int, 1> qaux_slice_sizes_;
  // number of blocks at each momentum
  std::vector<size_t> kao_nblocks_;
  std::vector<size_t> qaux_nblocks_;

  // store only ink
  // sizes of each irrep
  std::vector<tensor<int, 1> > ao_sizes_irre_;  // (ik, sizes)
  std::vector<tensor<int, 1> > aux_sizes_irre_;  // (ik, sizes)
  // offsets in flattened vector
  std::vector<tensor<int, 1> > ao_offsets_irre_;  // (ik, offsets)
  std::vector<tensor<int, 1> > aux_offsets_irre_;  // (ik, offsets)
  // offsets in block matrix, used for testing purpose only
  std::vector<tensor<int, 1> > ao_block_offsets_irre_;  // (ik, offsets)
  std::vector<tensor<int, 1> > aux_block_offsets_irre_;  // (ik, offsets)
  // start index of slices of different k points
  tensor<int, 1> kao_slice_offsets_irre_;
  tensor<int, 1> qaux_slice_offsets_irre_;
  tensor<int, 1> kao_slice_sizes_irre_;
  tensor<int, 1> qaux_slice_sizes_irre_;

  // start index of slices in interaction tensor
  tensor<long, 1> kpair_slice_offsets_;
  tensor<long, 1> kpair_slice_sizes_;
  std::vector<tensor<int, 1> > tensor_offsets_;  // (kpair, offsets)
  std::vector<tensor<int, 2> > tensor_irreps_;  // (kpair, irreps)
  std::vector<size_t> kpair_nblocks_;

  size_t max_kao_size_ = 0;  // maximum size of flattened ao space matrix at different k points
  size_t max_qaux_size_ = 0;  // maximum size of flattened aux space matrix at different q points
  size_t max_kpair_size_ = 0;  // maximum size of flattened interaction tensor

  // rotation matrices
  tensor<dcomplex, 1> kspace_orep_;  // (nk, flat_size)
  // k space representation matrix for all k points with one chosen operation in auxiliary basis space
  // (decomposed auxiliary basis)
  tensor<dcomplex, 1> kspace_auxrep_trans_;  // (nk, flat_size)
  // symmetry operations that transform k points in iBZ to full BZ. Dimension: ith k point, jth op
  std::vector<std::vector<long> > kpts_ops_;

  tensor<int, 1> korep_slice_offsets_;
  tensor<int, 1> korep_slice_sizes_;
  tensor<int, 1> kauxrep_slice_offsets_;
  tensor<int, 1> kauxrep_slice_sizes_;
  std::vector<tensor<int, 2> > orep_irreps_;  // (k, irreps)
  std::vector<tensor<int, 1> > orep_offsets_;  // (k, offsets)
  std::vector<tensor<int, 2> > orep_sizes_;  // (k, sizes)
  std::vector<tensor<int, 2> > auxrep_irreps_;  // (k, irreps)
  std::vector<tensor<int, 1> > auxrep_offsets_;  // (k, offsets)
  std::vector<tensor<int, 2> > auxrep_sizes_;  // (k, sizes)

  void set_orep_info(tensor<dcomplex, 1> &orep, tensor<int, 1> &slice_offsets, tensor<int, 1> &slice_sizes,
                     std::vector<tensor<int, 2> > &irreps,
                     std::vector<tensor<int, 1> > &offsets, std::vector<tensor<int, 2> > &sizes,
                     alps::hdf5::archive &in_file,
                     const std::string &group, const std::string &name);
  void set_identity_orep_info(tensor<dcomplex, 1> &orep, tensor<int, 1> &slice_offsets, tensor<int, 1> &slice_sizes,
                              std::vector<tensor<int, 2> > &irreps,
                              std::vector<tensor<int, 1> > &offsets, std::vector<tensor<int, 2> > &sizes,
                              const std::vector<tensor<int, 1> > &block_sizes,
                              const std::vector<tensor<int, 1> > &block_offsets);
  template<typename T>
  void apply_time_reversal(tensor<T, 1> &vec) {
    for (int k = 0; k < nk_; ++k) {
      int ck = conj_index_[k];
      if (ck != k) { vec(k) = vec(ck); }
    }
  }
  template<typename T>
  void apply_time_reversal(std::vector<T> &vec) {
    for (int k = 0; k < nk_; ++k) {
      int ck = conj_index_[k];
      if (ck != k) { vec[k] = vec[ck]; }
    }
  }

public:

  static int find_pos(const tensor<double, 1> &k, const tensor<double, 2> &kmesh) {
    for (size_t i = 0; i < kmesh.shape()[0]; ++i) {
      bool found = true;
      for (size_t j = 0; j < k.shape()[0]; ++j) {
        found &= std::abs(k(j) - kmesh(i, j)) < 1e-9;
      }
      if (found) {
        return int(i);
      }
    }
    throw std::logic_error(
      "K point (" + std::to_string(k(0)) + ", " + std::to_string(k(1)) + ", " + std::to_string(k(2)) +
      ") has not been found in the mesh.");
  }

  static tensor<double, 1> wrap_1stBZ(const tensor<double, 1> &k) {
    tensor<double, 1> kk = k;
    for (size_t j = 0; j < kk.shape()[0]; ++j) {
      while ((kk(j) - 9.9999999999e-1) > 0.0) {
        kk(j) -= 1.0;
      }
      if (std::abs(kk(j)) < 1e-9) {
        kk(j) = 0.0;
      }
      while (kk(j) < 0) {
        kk(j) += 1.0;
      }
    }
    return kk;
  };
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_SYMMETRY_UTILS_H
