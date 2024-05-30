
#include "symmetry_utils.h"

namespace symmetry_mbpt {

symmetry_utils_t::symmetry_utils_t(const std::string &dfintegral_file,
                                   size_t nk,
                                   const std::string &symm_file_name,
                                   bool read_kpair,
                                   const std::string &rot_file_name,
                                   bool rotate, bool time_reversal): nk_(nk), nkpw_(1./double(nk_)),
                                   q_ind_(nk_, nk_), q_ind2_(nk_, nk_),
                                   rotate_(rotate), time_reversal_(time_reversal) {

  kmesh_.reshape(nk_, 3);
  std::string path = dfintegral_file + "/meta.h5";
  alps::hdf5::archive int_file(path, "r");
  int_file["grid/num_kpair_stored"] >> num_kpair_stored_;
  int_file["grid/k_mesh_scaled"] >> kmesh_;
  int_file.close();

  tensor<double, 2> qmesh(kmesh_.shape());
  for (int j = 0; j < nk_; ++j) {
    auto ki = kmesh_(0);
    auto kj = kmesh_(j);
    auto kq = wrap_1stBZ(ki - kj);
    qmesh(j) = kq;
  }

  for (int i = 0; i < nk_; ++i) {
    auto ki = kmesh_(i);
    for (int j = 0; j < nk_; ++j) {
      auto kj = kmesh_(j);
      auto kq = wrap_1stBZ(ki - kj);
      int q = find_pos(kq, qmesh);
      q_ind_(j, q) = i;
      q_ind2_(i, j) = q;
    }
  }

  // -- rotation related indices
  index_.resize(nk_);
  irre_index_.resize(nk_);
  if ((!rotate_) || rot_file_name.empty()) {
    rotate_ = false;
    ink_ = nk_;
    irre_list_.resize(ink_);
    weight_.resize(ink_);
    std::iota(irre_list_.begin(), irre_list_.end(), 0);
    std::fill(weight_.begin(), weight_.end(), 1);
    std::iota(index_.begin(), index_.end(), 0);
    std::iota(irre_index_.begin(), irre_index_.end(), 0);
  } else {
    alps::hdf5::archive in_file(rot_file_name, "r");
    in_file["symmetry/KStruct/k_ibz_index"] >> irre_list_;
    in_file["symmetry/KStruct/irr_list"] >> index_;
    in_file["symmetry/KStruct/irre_index"] >> irre_index_;
    in_file["symmetry/KStruct/weight"] >> weight_;
    in_file["symmetry/n_star"] >> ink_;
    in_file.close();
  }

  if (time_reversal_ && (!rot_file_name.empty())) {
    alps::hdf5::archive in_file(rot_file_name, "r");
    // need to match the condition we use when generating integrals
    in_file["symmetry/time_reversal"] >> time_reversal_;
    in_file["symmetry/KStruct/irre_conj_list"] >> irre_conj_list_;
    in_file["symmetry/KStruct/conj_index"] >> conj_index_;
    in_file["symmetry/KStruct/irre_conj_index"] >> irre_conj_index_;
    in_file["symmetry/KStruct/irre_conj_weight"] >> irre_conj_weight_;
    in_file["symmetry/n_conj_star"] >> ink_;
    in_file.close();
  } else {
    time_reversal_ = false;
    conj_index_.resize(nk_);
    irre_conj_index_.resize(nk_);
    irre_conj_list_.resize(ink_);
    irre_conj_weight_.resize(ink_);

    std::iota(conj_index_.begin(), conj_index_.end(), 0);
    irre_conj_index_ = irre_index_;
    irre_conj_list_ = irre_list_;
    irre_conj_weight_ = weight_;
  }

  k_pairs_.reshape(ink_*nk_, 3);
  for (int i = 0; i < ink_; ++i) {
    for (int j = 0; j < nk_; ++j) {
      int index = i * nk_ + j;
      k_pairs_(index, 0) = i;
      k_pairs_(index, 1) = irre_conj_list_[i];
      k_pairs_(index, 2) = j;
    }
  }

  // -- read in block diagonalization related indices
  if (!symm_file_name.empty()) {
    kao_slice_offsets_.reshape(nk);
    qaux_slice_offsets_.reshape(nk);
    kao_slice_sizes_.reshape(nk);
    qaux_slice_sizes_.reshape(nk);
    kao_nblocks_.resize(nk);
    qaux_nblocks_.resize(nk);

    ao_sizes_.resize(nk);
    aux_sizes_.resize(nk);
    ao_offsets_.resize(nk);
    aux_offsets_.resize(nk);

    ao_block_offsets_.resize(nk);
    aux_block_offsets_.resize(nk);

    alps::hdf5::archive symm_file(symm_file_name, "r");

    symm_file["block/kao_slice_offsets"] >> kao_slice_offsets_;
    symm_file["block/qaux_slice_offsets"] >> qaux_slice_offsets_;
    symm_file["block/kao_slice_sizes"] >> kao_slice_sizes_;
    symm_file["block/qaux_slice_sizes"] >> qaux_slice_sizes_;
    symm_file["block/kao_nblocks"] >> kao_nblocks_;
    symm_file["block/qaux_nblocks"] >> qaux_nblocks_;

    for (int i = 0; i < nk_; ++i) {
      int ck = conj_index_[i];
      ao_sizes_[i].reshape(kao_nblocks_[i]);
      ao_offsets_[i].reshape(kao_nblocks_[i]);
      ao_block_offsets_[i].reshape(kao_nblocks_[i]);
      aux_sizes_[i].reshape(qaux_nblocks_[i]);
      aux_offsets_[i].reshape(qaux_nblocks_[i]);
      aux_block_offsets_[i].reshape(qaux_nblocks_[i]);
      symm_file["block/ao_sizes/" + std::to_string(ck)] >> ao_sizes_[i];
      symm_file["block/ao_offsets/" + std::to_string(ck)] >> ao_offsets_[i];
      symm_file["block/ao_block_offsets/" + std::to_string(ck)] >> ao_block_offsets_[i];
      symm_file["block/aux_sizes/" + std::to_string(ck)] >> aux_sizes_[i];
      symm_file["block/aux_offsets/" + std::to_string(ck)] >> aux_offsets_[i];
      symm_file["block/aux_block_offsets/" + std::to_string(ck)] >> aux_block_offsets_[i];
    }
    // compute max_kao_size and max_qaux_size
    max_kao_size_ = kao_slice_sizes_.vector().maxCoeff();
    max_qaux_size_ = qaux_slice_sizes_.vector().maxCoeff();

    if (read_kpair) {
      kpair_slice_offsets_.reshape(num_kpair_stored_);
      kpair_slice_sizes_.reshape(num_kpair_stored_);
      kpair_nblocks_.resize(num_kpair_stored_);
      tensor_offsets_.resize(num_kpair_stored_);
      tensor_irreps_.resize(num_kpair_stored_);
      symm_file["block/kpair_slice_offsets"] >> kpair_slice_offsets_;
      symm_file["block/kpair_slice_sizes"] >> kpair_slice_sizes_;
      symm_file["block/kpair_nblocks"] >> kpair_nblocks_;
      for (int i = 0; i < num_kpair_stored_; ++i) {
        tensor_offsets_[i].reshape(kpair_nblocks_[i]);
        tensor_irreps_[i].reshape({kpair_nblocks_[i], 3});
        symm_file["block/tensor_offsets/" + std::to_string(i)] >> tensor_offsets_[i];
        symm_file["block/tensor_irreps/" + std::to_string(i)] >> tensor_irreps_[i];
      }
      // compute max_kpair_size
      max_kpair_size_ = kpair_slice_sizes_.vector().maxCoeff();
    }
    symm_file.close();

    // -- iBZ related
    kao_slice_offsets_irre_.reshape(ink_);
    qaux_slice_offsets_irre_.reshape(ink_);
    kao_slice_sizes_irre_.reshape(ink_);
    qaux_slice_sizes_irre_.reshape(ink_);

    ao_sizes_irre_.reserve(ink_);
    aux_sizes_irre_.reserve(ink_);
    ao_offsets_irre_.reserve(ink_);
    aux_offsets_irre_.reserve(ink_);
    ao_block_offsets_irre_.reserve(ink_);
    aux_block_offsets_irre_.reserve(ink_);

    int k_offsets = 0;
    int q_offsets = 0;
    for (int ik = 0; ik < ink_; ++ik) {
      size_t k = irre_conj_list_[ik];
      kao_slice_sizes_irre_(ik) = kao_slice_sizes_(k);
      qaux_slice_sizes_irre_(ik) = qaux_slice_sizes_(k);
      kao_slice_offsets_irre_(ik) = k_offsets;
      qaux_slice_offsets_irre_(ik) = q_offsets;
      k_offsets += kao_slice_sizes_irre_(ik);
      q_offsets += qaux_slice_sizes_irre_(ik);

      ao_sizes_irre_.emplace_back(ao_sizes_[k]);
      aux_sizes_irre_.emplace_back(aux_sizes_[k]);
      ao_offsets_irre_.emplace_back(ao_offsets_[k]);
      aux_offsets_irre_.emplace_back(aux_offsets_[k]);
      ao_block_offsets_irre_.emplace_back(ao_block_offsets_[k]);
      aux_block_offsets_irre_.emplace_back(aux_block_offsets_[k]);
    }
  }

  /*
   * get k space representations with 1st operation in operation list
   */
  if (rotate_ && (!rot_file_name.empty())) {
    alps::hdf5::archive in_file(rot_file_name, "r");
    kpts_ops_.resize(nk_);
    for (int i = 0; i < nk_; ++i) {
      in_file["symmetry/Kpoint/"+std::to_string(i)+"/ops"] >> kpts_ops_[i];
    }
    set_orep_info(kspace_orep_, korep_slice_offsets_, korep_slice_sizes_,
                  orep_irreps_, orep_offsets_, orep_sizes_,
                  in_file, "/kao_rot", "orep");
    set_orep_info(kspace_auxrep_trans_, kauxrep_slice_offsets_, kauxrep_slice_sizes_,
                  auxrep_irreps_, auxrep_offsets_, auxrep_sizes_,
                  in_file, "/kaux_rot", "auxrep");
    in_file.close();
  } else if (!symm_file_name.empty()) {
    // fake identity matrices, need block info
    set_identity_orep_info(kspace_orep_, korep_slice_offsets_, korep_slice_sizes_,
                           orep_irreps_, orep_offsets_, orep_sizes_, ao_sizes_, ao_offsets_);
    set_identity_orep_info(kspace_auxrep_trans_, kauxrep_slice_offsets_, kauxrep_slice_sizes_,
                           auxrep_irreps_, auxrep_offsets_, auxrep_sizes_,
                           aux_sizes_, aux_offsets_);
  }
}

void symmetry_utils_t::set_orep_info(tensor<dcomplex, 1> &orep,
                                     tensor<int, 1> &slice_offsets, tensor<int, 1> &slice_sizes,
                                     std::vector<tensor<int, 2> > &irreps,
                                     std::vector<tensor<int, 1> > &offsets, std::vector<tensor<int, 2> > &sizes,
                                     alps::hdf5::archive &in_file,
                                     const std::string &group, const std::string &name) {

  tensor<dcomplex, 2> orep_temp;
  tensor<int, 3> irreps_temp;
  tensor<int, 2> offsets_temp;
  tensor<int, 3> sizes_temp;
  std::vector<tensor<dcomplex, 1> > orep_vec;
  orep_vec.reserve(nk_);
  irreps.reserve(nk_);
  offsets.reserve(nk_);
  sizes.reserve(nk_);
  int slice_offset = 0;
  slice_offsets.reshape(nk_);
  slice_sizes.reshape(nk_);
  for (int k = 0; k < nk_; ++k) {
    auto ck = conj_index_[k]; // k is considered as ck when doing rotation
    auto ik = index_[ck]; // corresponding index of k point in iBZ
    auto op = kpts_ops_[ck][0];
    in_file[group+"/"+name+"_flat/"+std::to_string(ik)] >> orep_temp;
    in_file[group+"/"+name+"_irreps/"+std::to_string(ik)] >> irreps_temp;
    in_file[group+"/"+name+"_offsets/"+std::to_string(ik)] >> offsets_temp;
    in_file[group+"/"+name+"_sizes/"+std::to_string(ik)] >> sizes_temp;
    orep_vec.emplace_back(orep_temp(op));
    irreps.emplace_back(irreps_temp(op));
    offsets.emplace_back(offsets_temp(op));
    sizes.emplace_back(sizes_temp(op));
    int slice_size = offsets_temp(op)(offsets_temp(op).size()-1)
        + sizes_temp(op)(sizes_temp(op).shape()[0]-1)(0)*sizes_temp(op)(sizes_temp(op).shape()[0]-1)(1);
    slice_offsets(k) = slice_offset;
    slice_offset += slice_size;
    slice_sizes(k) = slice_size;
  }
  orep.reshape(slice_offsets(nk_-1)+slice_sizes(nk_-1));
  for (int k = 0; k < nk_; ++k) {
    Mcolumn<dcomplex>(orep.data()+slice_offsets(k), slice_sizes(k)) = orep_vec[k].vector();
  }
}

void symmetry_utils_t::set_identity_orep_info(tensor<dcomplex, 1> &orep,
                                              tensor<int, 1> &slice_offsets, tensor<int, 1> &slice_sizes,
                                              std::vector<tensor<int, 2> > &irreps,
                                              std::vector<tensor<int, 1> > &offsets,
                                              std::vector<tensor<int, 2> > &sizes,
                                              const std::vector<tensor<int, 1> > &block_sizes,
                                              const std::vector<tensor<int, 1> > &block_offsets) {
  std::vector<tensor<dcomplex, 1> > orep_vec;
  orep_vec.reserve(nk_);
  irreps.reserve(nk_);
  offsets.reserve(nk_);
  sizes.reserve(nk_);
  tensor<dcomplex, 1> orep_temp;
  tensor<int, 2> irreps_temp;
  tensor<int, 2> sizes_temp;
  int slice_offset = 0;
  slice_offsets.reshape(nk_);
  slice_sizes.reshape(nk_);
  for (int k = 0; k < nk_; ++k) {
    int n_block = block_sizes[k].size();
    irreps_temp.reshape(n_block, 2);
    sizes_temp.reshape(n_block, 2);
    int slice_size = block_offsets[k](n_block-1) + block_sizes[k](n_block-1) * block_sizes[k](n_block-1);
    slice_offsets(k) = slice_offset;
    slice_offset += slice_size;
    slice_sizes(k) = slice_size;
    orep_temp.reshape(slice_size);
    for (int i = 0; i < n_block; ++i) {
      irreps_temp(i, 0) = i;
      irreps_temp(i, 1) = i;
      sizes_temp(i, 0) = block_sizes[k](i);
      sizes_temp(i, 1) = block_sizes[k](i);
      MMatrixX<dcomplex>(orep_temp.data()+block_offsets[k](i), block_sizes[k](i), block_sizes[k](i))
          = Matrix<dcomplex>::Identity(block_sizes[k](i), block_sizes[k](i));
    }
    orep_vec.emplace_back(orep_temp);
    irreps.emplace_back(irreps_temp);
    sizes.emplace_back(sizes_temp);
    offsets.emplace_back(block_offsets[k]);
  }
  orep.reshape(slice_offsets(nk_-1)+slice_sizes(nk_-1));
  for (int k = 0; k < nk_; ++k) {
    Mcolumn<dcomplex>(orep.data()+slice_offsets(k), slice_sizes(k)) = orep_vec[k].vector();
  }
}

/**
  * compute integral momenta using momentum conservation of the first integral
  *
  * @param n - first integral momentum triplet
  * @return full 4 momenta for the current triplet
  */
std::array<size_t, 4> symmetry_utils_t::momentum_conservation(const std::array<size_t, 3> &n,
                                                              IntegralType type) const {
  std::array<size_t, 4> k{0};
  if (type == first_integral) {
    k[0] = n[0];
    k[1] = n[1];
    k[2] = n[2];
    k[3] = mom_cons(n[0], n[1], n[2]);
  } else if (type == second_direct) {
    k[0] = n[1];
    k[1] = n[0];
    k[2] = mom_cons(n[0], n[1], n[2]);
    k[3] = n[2];
  } else {
    k[0] = mom_cons(n[0], n[1], n[2]);
    k[1] = n[0];
    k[2] = n[1];
    k[3] = n[2];
  }
  return k;
}

size_t symmetry_utils_t::mom_cons(size_t i, size_t j, size_t k) const {
  int q = q_ind2_(i, j);
  int l = q_ind_(k, q);
  return l;
}

} // namespace symmetry_mbpt
