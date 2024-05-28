
#include "utils.h"
#include "k_mesh.h"

namespace symmetry_adaptation {

void KMesh::read_k_mesh(const std::string &h5file, double tol) {
  Ndarray<double> kmesh(std::pow(q_, DIM), DIM);
  alps::hdf5::archive ar(h5file);

  ar["/k_mesh"] >> kmesh;
  for (int i = 0; i < kpts_.size(); ++i) {
    kpts_[i] = Ndarray_MatView(kmesh, std::pow(q_, DIM), DIM).row(i);
    wrap_kpt_to_BZ(kpts_[i], trans_vec_);
  }
  ar["/k_mesh_scaled"] >> kmesh;
  for (int i = 0; i < kpts_.size(); ++i) {
    kpts_scaled_[i] = Ndarray_MatView(kmesh, std::pow(q_, DIM), DIM).row(i);
    wrap_kpt_scaled_to_BZ(kpts_scaled_[i]);
  }
  find_self_conjugate_kpts(tol);
}

void KMesh::generate_k_mesh(const std::string &convention, double tol) {

  const auto &t = trans_vec_;
  ColVector<double> k_lin;
  if (convention == "pyscf") {
    k_lin = Eigen::VectorXd::LinSpaced(q_, 0, q_ - 1).array() / q_;
    wrap_kpt_scaled_to_BZ(k_lin);
  }
  else if (convention == "monkhorst") {
    for (int r = 0; r < q_; ++r) {
      k_lin(r) = (2. * r - q_) / (2. * q_);
    }
  }
  else {
    throw std::runtime_error("k mesh convention undefined");
  }

  kpts_.reserve(std::pow(q_, DIM));
  kpts_scaled_.reserve(std::pow(q_, DIM));
  for (int r1 = 0; r1 < q_; ++r1) {
    for (int r2 = 0; r2 < q_; ++r2) {
      for (int r3 = 0; r3 < q_; ++r3) {
        kpts_.emplace_back(k_lin[r1] * t.b(0) + k_lin[r2] * t.b(1) + k_lin[r3] * t.b(2));
        kpts_scaled_.emplace_back(ColVector<double, 3>{k_lin[r1], k_lin[r2], k_lin[r3]});
      }
    }
  }
  find_self_conjugate_kpts(tol);
}

int KMesh::find_k_index(const std::vector<ColVector<double, DIM> > &kmesh,
                        const ColVector<double> &kpt, double tol) {
  for (int i = 0; i < kmesh.size(); ++i) {
    if ((kpt - kmesh[i]).norm() < tol) {
      return i;
    }
  }
  throw std::runtime_error("Can not find k point in k mesh");
}

void KMesh::find_self_conjugate_kpts(double tol) {
  self_conjugate_.resize(kpts_.size());
  ColVector<double, DIM> kpt_conj;
  for (int i = 0; i < kpts_.size(); ++i) {
    const auto &kpt = kpts_scaled_[i];
    kpt_conj = -kpt;
    wrap_kpt_scaled_to_BZ(kpt_conj);
    if ((kpt - kpt_conj).norm() < tol)
        self_conjugate_[i] = true;
    else
      self_conjugate_[i] = false;
  }
}

void KMesh::save(alps::hdf5::archive &ar, const std::string &group) const {
  std::string prefix = group + "/grid/";

  Ndarray<double> kpts_temp(kpts_.size(), DIM);
  Ndarray<double> kpts_temp_scaled(kpts_.size(), DIM);

  for (int i = 0; i < kpts_.size(); ++i) {
    Ndarray_VecView(kpts_temp, DIM, i*DIM) = kpts_[i];
    Ndarray_VecView(kpts_temp_scaled, DIM, i*DIM) = kpts_scaled_[i];
  }

  ar[prefix + "k_mesh"] << kpts_temp;
  ar[prefix + "k_mesh_scaled"] << kpts_temp_scaled;
}

} // namespace symmetry_adaptation