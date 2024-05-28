
#include "k_space_structure.h"

namespace symmetry_adaptation {

void KSpaceStructure::generate_transform_table(bool verbose) {

  if(verbose) {
    std::cout << "generating k space transform table." << std::endl;
  }

  transform_table_.resize(k_mesh_.size(), group_.order());
  transform_table_rev_.resize(k_mesh_.size(), group_.order());

  ColVector<double, DIM> transformed_vec;
  for (int i = 0; i < k_mesh_.size(); ++i) {
    for (int g = 0; g < group_.order(); ++g) {
      transformed_vec = group_.reciprocal_space_rep(g) * k_mesh_.kpt(i);
      //std::cout << "k space vector before wrapping: " << transformed_vec.transpose() << std::endl;
      KMesh::wrap_kpt_to_BZ(transformed_vec, trans_vec_);
      //std::cout << "k space vector after wrapping: " << transformed_vec.transpose() << std::endl;
      bool found = false;
      for (int j = 0; j < k_mesh_.size(); ++j) {
        if ((transformed_vec - k_mesh_.kpt(j)).norm() < tol_) {
          transform_table_(i, g) = j;
          transform_table_rev_(j, g) = i;
          found = true;
          break;
        }
      }
      if (!found) {
        std::cout << "origin vector: " << k_mesh_.kpt(i).transpose() << std::endl;
        std::cout << "transformed vector: " << transformed_vec.transpose() << std::endl;
        std::cout << "operation: " << std::endl;
        std::cout << group_.reciprocal_space_rep(g) << std::endl;
        throw std::runtime_error(
          "group op " + std::to_string(g) + " transformed k-pt to new k-point, violating symmetry");
      }
    }
  }

  if (verbose) {
    std::cout << "transformation table in k space:" << std::endl;
    std::cout << transform_table_ << std::endl;
  }
}

void KSpaceStructure::find_stars() {

  int n = k_mesh_.size();
  // List of all k points
  ColVector<int> a = ColVector<int>::LinSpaced(Eigen::Sequential, n, 0, n-1);
  std::set<int> kpoints(a.data(), a.data()+n);
  while (!kpoints.empty()) {
    int k = *kpoints.begin();
    auto star = std::set<int>(transform_table_.row(k).data(), transform_table_.row(k).data()+transform_table_.cols());
    star.insert(k);
    std::vector<ColVector<double, DIM> > star_k;
    std::vector<int> star_idx;
    for (int i : star) {
      kpoints.erase(i);
      star_k.push_back(k_mesh_.kpt(i));
      star_idx.push_back(i);
    }
    stars_.push_back(star_k);
    stars_index_.push_back(star_idx);
    weights_.push_back(star_k.size());
  }
}

void KSpaceStructure::find_iBZ(bool verbose) {

  auto compare = [](const Eigen::Vector3d& s1, const Eigen::Vector3d& s2) {
    double tol = 1e-4;
    auto diff = s1 - s2;
    bool gre = std::make_tuple(s1(0), s1(1), s1(2)) > std::make_tuple(s2(0), s2(1), s2(2));
    bool les = std::make_tuple(diff(0), diff(1), diff(2)) < std::make_tuple(tol, tol, tol);
    return (!les) || gre;
  };

  stars_rep_.reserve(n_star());
  stars_ops_.reserve(n_star());
  stars_rep_index_.reserve(n_star());
  ks_ops_.resize(k_mesh_.size());
  for(int s = 0; s < stars_.size(); ++s) {
    auto &star = stars_[s];
    std::sort(star.begin(), star.end(), compare);
    std::reverse(star.begin(), star.end());

    stars_rep_.emplace_back(star[0]);
    stars_rep_index_.emplace_back(k_mesh_.find_k_index(k_mesh_.kpts(), star[0]));
    if (verbose) {
      std::cout << "star of size: " << star.size() << " rep: " << star[0].transpose() << std::endl;
    }
    stars_ops_.emplace_back(find_symm_op(star));
    for (int i = 0; i < star.size(); ++i) {
      stars_index_[s][i] = k_mesh_.find_k_index(k_mesh_.kpts(), star[i]);
      ks_ops_[stars_index_[s][i]] = stars_ops_[s][i];
    }
  }
}

std::vector<std::vector<int> > KSpaceStructure::find_symm_op(std::vector<ColVector<double, DIM> > &star) {

  int size = star.size();
  std::vector< std::vector<int> > ops(size);
  const auto &rep = star[0];

  int op_nums = 0;
  ColVector<double, 3> trans_k;

  for(int i = 0; i < size; ++i) {
    const auto &s = star[i];
    std::vector<int> op;
    for(int g = 0; g < group_.h(); ++g){
      trans_k = group_.reciprocal_space_rep(g) * rep;
      KMesh::wrap_kpt_to_BZ(trans_k, trans_vec_);
      if ((s - trans_k).norm() < tol_) {
        op.push_back(g);
        op_nums += 1;
      }
    }
    if (op.empty())
      throw std::runtime_error("can not find symmetry operations for k points in a star");

    ops[i] = op;
  }
  if (op_nums != group_.h()) {
    std::cout << "operation numbers: " << op_nums << std::endl;
    std::cout << "number of group elements: " << group_.h() << std::endl;
    throw std::runtime_error("symmetry operations in stars break");
  }
  return ops;
}

void KSpaceStructure::find_ir_list() {

  ir_list_.resize(k_mesh_.size());
  irre_index_.resize(k_mesh_.size());
  for (int s = 0; s < n_star(); ++s) {
    for (int x = 0; x < stars_[s].size(); ++x) {
      ir_list_[stars_index_[s][x]] = stars_rep_index_[s];
      irre_index_[stars_index_[s][x]] = s;
    }
  }
}

void KSpaceStructure::find_conjugate_relation() {

  if (!time_reversal_) {
    stars_conj_index_.resize(n_star());
    std::iota(stars_conj_index_.begin(), stars_conj_index_.end(), 0);

    conj_index_.resize(nk_);
    std::iota(conj_index_.begin(), conj_index_.end(), 0);

    conj_list_ = conj_index_;
    conj_check_.resize(nk_);
    std::fill(conj_check_.begin(), conj_check_.end(), 0);
  }
  else {
    stars_conj_index_.resize(n_star());
    std::fill(stars_conj_index_.begin(), stars_conj_index_.end(), -1);
    std::vector<int> kidx(nk_);
    for (int i = 0; i < n_star(); ++i) {
      if (stars_conj_index_[i] != -1) continue;
      stars_conj_index_[i] = i;
      const auto &ks = stars_index_[i];
      std::vector<int> k_rem_idx;
      std::set_difference(kidx.begin(), kidx.end(),
                          ks.begin(), ks.end(),
                          std::back_inserter(k_rem_idx));
      std::vector<ColVector<double, DIM> > k_rem(k_rem_idx.size());
      for (int j = 0; j < k_rem_idx.size(); ++j) {
        k_rem[j] = k_mesh_.kpts_scaled()[k_rem_idx[j]];
      }
      int rep_idx = star_rep_index(i);
      ColVector<double, DIM> k_conj = -k_mesh_.kpts_scaled()[rep_idx];
      KMesh::wrap_kpt_scaled_to_BZ(k_conj);
      try {
        int k_conj_idx = k_mesh_.find_k_index(k_rem, k_conj);
        int k_conj_rep_idx = ir_list_[k_conj_idx];
        int conj_star_idx = std::find(stars_rep_index_.begin(), stars_rep_index_.end(),
                                      k_conj_rep_idx) - stars_rep_index_.begin();
        stars_conj_index_[conj_star_idx] = i;
      }
      catch(const std::runtime_error &e) {
        continue;
      }
    }

    // -- find k conj list
    auto rep_stars = std::set<int>(stars_conj_index_.begin(), stars_conj_index_.end());
    std::vector<int> rep_stars_k_idx;
    for (int i: rep_stars) {
        std::copy(stars_index_[i].begin(), stars_index_[i].end(), std::back_inserter(rep_stars_k_idx));
    }
    std::sort(rep_stars_k_idx.begin(), rep_stars_k_idx.end());

    conj_index_.resize(nk_);
    std::iota(conj_index_.begin(), conj_index_.end(), 0);
    for (int k_idx = 0; k_idx < nk_; ++k_idx) {
      ColVector<double, DIM> k_conj = -k_mesh_.kpts_scaled()[k_idx];
      KMesh::wrap_kpt_scaled_to_BZ(k_conj);
      int k_conj_idx = k_mesh_.find_k_index(k_mesh_.kpts_scaled(), k_conj);

      bool k_flag = std::find(rep_stars_k_idx.begin(), rep_stars_k_idx.end(), k_idx) != rep_stars_k_idx.end();
      bool k_conj_flag = std::find(rep_stars_k_idx.begin(), rep_stars_k_idx.end(), k_conj_idx) != rep_stars_k_idx.end();
      if (k_flag && k_conj_flag) {
        if (k_idx <= k_conj_idx)
          conj_index_[k_conj_idx] = k_idx;
        else
          conj_index_[k_idx] = k_conj_idx;
      }
      else if (k_flag)
        conj_index_[k_conj_idx] = k_idx;
      else if (k_conj_flag)
        conj_index_[k_idx] = k_conj_idx;
      else
        throw std::runtime_error("Find k conj list fail");
    }

    auto conj_list = std::set<int>(conj_index_.begin(), conj_index_.end());
    conj_list_.resize(conj_list.size());
    std::copy(conj_list.begin(), conj_list.end(), conj_list_.begin());

    conj_check_.resize(nk_);
    std::fill(conj_check_.begin(), conj_check_.end(), 0);
    for (int i = 0; i < conj_index_.size(); ++i) {
      if (conj_index_[i] != i)
        conj_check_[i] = 1;
    }
  }

  auto unique_index = std::set<int>(stars_conj_index_.begin(), stars_conj_index_.end());
  irre_conj_list_.reserve(unique_index.size());
  for (int i : unique_index) {
    irre_conj_list_.emplace_back(star_rep_index(i));
  }

  irre_conj_index_.reserve(nk_);
  irre_conj_weight_.resize(irre_conj_list_.size());
  std::fill(irre_conj_weight_.begin(), irre_conj_weight_.end(), 0);
  for (int i = 0; i < nk_; ++i) {
    int i_conj = conj_index_[i];
    int ik = ir_list_[i_conj];
    int conj_idx = std::find(irre_conj_list_.begin(), irre_conj_list_.end(), ik) - irre_conj_list_.begin();
    irre_conj_index_.emplace_back(conj_idx);
    irre_conj_weight_[conj_idx] += 1;
  }
}

void KSpaceStructure::find_little_cogroup(bool verbose) {

  if (verbose) {
    std::cout << "little cogroup of k points:" << std::endl;
  }

  int size = k_mesh_.kpts().size();
  little_cogroup_.resize(size);

  int op_nums = 0;
  ColVector<double, 3> trans_k;

  for(int i = 0; i < size; ++i) {
    const auto &k = k_mesh_.kpt(i);
    std::vector<int> op;
    for(int g = 0; g < group_.h(); ++g){
      trans_k = group_.reciprocal_space_rep(g) * k;
      KMesh::wrap_kpt_to_BZ(trans_k, trans_vec_);
      if ((k - trans_k).norm() < tol_) {
        op.push_back(g);
        op_nums += 1;
      }
    }
    if (op.empty())
      throw std::runtime_error("can not find symmetry operations for k point" + std::to_string(i));
    little_cogroup_[i] = op;
    if (verbose) {
      std::cout << "little cogroup of k point " << i << ": " << std::endl;
      for (const auto &x: little_cogroup_[i]) {
        std::cout << x << " ";
      }
      std::cout << std::endl;
    }
  }

  /*
   * Little cogroup of star representatives
   */
  if (verbose) {
    std::cout << "little cogroup of stars:" << std::endl;
  }

  star_little_cogroup_.reserve(stars_.size());
  for (int s = 0; s < stars_.size(); ++s) {
    star_little_cogroup_.emplace_back(stars_ops_[s][0]);
    if (verbose) {
      std::cout << "little cogroup of star " << s << ": " << std::endl;
      for (const auto &x: star_little_cogroup_[s]) {
        std::cout << x << " ";
      }
      std::cout << std::endl;
    }
  }
}

void KSpaceStructure::find_little_cogroup_conjugacy_classes(bool verbose) {

  /*
   * Little cogroup conjugacy classes of all k points
   */

  if (verbose) {
    std::cout << "conjugacy classes of little cogroups:" << std::endl;
  }

  int size = k_mesh_.kpts().size();

  little_cogroup_conjugacy_classes_.resize(size);
  for (int k = 0; k < size; ++k) {
    std::vector<bool> conj_el_found(group_.h(), false);
    int n = 0;
    for (int g: little_cogroup_[k]) {
      if (conj_el_found[g]) continue;
      std::vector<int> new_conj_class;
      for (int a: little_cogroup_[k]) {
        int againv = group_.multiplication_table()(group_.multiplication_table()(a, g), group_.operation_inverse(a));
        if (!conj_el_found[againv]) {
          conj_el_found[againv] = true;
          new_conj_class.push_back(againv);
        }
      }
      little_cogroup_conjugacy_classes_[k].push_back(new_conj_class);
      n += new_conj_class.size();
    }
    if (n != little_cogroup_[k].size()) {
      throw std::runtime_error("can not find conjugacy classes for little cogroup of k point " + std::to_string(k));
    }
    if (verbose) {
      std::cout << little_cogroup_conjugacy_classes_[k].size() << " cc classes of k point " << k << ":" << std::endl;
      for (const auto &cc: little_cogroup_conjugacy_classes_[k]) {
        for (const auto &ccel: cc) std::cout << ccel << " ";
        std::cout << std::endl;
      }
    }
  }

  /*
   * Little cogroup conjugacy classes of star representatives
   */

  int nstar = stars_.size();
  star_little_cogroup_conjugacy_classes_.resize(nstar);
  for (int i = 0; i < nstar; ++i) {
    star_little_cogroup_conjugacy_classes_[i] = little_cogroup_conjugacy_classes_[star_rep_index(i)];

    if (verbose) {
      std::cout << star_little_cogroup_conjugacy_classes_[i].size() << " cc classes of star " << i << ":" << std::endl;
      for (const auto &cc: star_little_cogroup_conjugacy_classes_[i]) {
        for (const auto &ccel: cc) std::cout << ccel << " ";
        std::cout << std::endl;
      }
    }
  }
}

void KSpaceStructure::find_little_cogroup_conjugacy_classes_mult(bool verbose) {
  // Find conjugacy classes according to \beta \gamma = \alpha \beta, with \alpha, \gamma \in C, \beta \in G
  // This is a cross check of class method find_little_cogroup_conjugacy_classes

  if (verbose) {
    std::cout << "conjugacy classes of little cogroups with multiplication relation:" << std::endl;
  }

  int size = k_mesh_.kpts().size();

  little_cogroup_conjugacy_classes_mult_.resize(size);
  for (int k = 0; k < size; ++k) {
    std::vector<bool> conj_el_found(group_.h(), false);
    int n = 0;
    for (int a: little_cogroup_[k]) {
      if (conj_el_found[a]) continue;
      std::vector<int> new_conj_class;
      new_conj_class.push_back(a);
      conj_el_found[a] = true;
      for (int b: little_cogroup_[k]) {
        bool found = false;
        for (int g: little_cogroup_[k]) {
          if (group_.multiplication_table()(a, b) == group_.multiplication_table()(b, g)) {
            found = true;
            if (!conj_el_found[g]) {
              conj_el_found[g] = true;
              new_conj_class.push_back(g);
            }
          }
        } // g
        if (!found) {
          throw std::runtime_error(
            "can not find conjugate element for operation " + std::to_string(b) + " in little co-group of k point " +
            std::to_string(k));
        }
      } // b
      little_cogroup_conjugacy_classes_mult_[k].push_back(new_conj_class);
      n += new_conj_class.size();
    } // a
    if (n != little_cogroup_[k].size()) {
      throw std::runtime_error("can not find conjugacy classes for little cogroup of k point " + std::to_string(k));
    }
    if (verbose) {
      std::cout << little_cogroup_conjugacy_classes_mult_[k].size() << " cc classes of k point " << k << ":" << std::endl;
      for (const auto &cc: little_cogroup_conjugacy_classes_mult_[k]) {
        for (const auto &ccel: cc) std::cout << ccel << " ";
        std::cout << std::endl;
      }
    }
  } // k
}

void KSpaceStructure::save(alps::hdf5::archive &ar, const std::string &group) const {

  // Save k space transformation table
  ar[group + "KStruct/transform_table"] << transform_table();
  ar[group + "KStruct/transform_table_rev"] << transform_table_rev();

  // Save k space symmetry information
  Ndarray<double> stars_rep(n_star(), DIM);
  for (int s = 0; s < n_star(); ++s) {
    Ndarray_VecView(stars_rep, DIM, s*DIM) = stars_rep_[s];
  }
  ar[group + "KStruct/k_ibz"] << stars_rep;
  ar[group + "KStruct/k_ibz_index"] << stars_rep_index_;
  ar[group + "KStruct/irr_list"] << ir_list_;
  ar[group + "KStruct/irre_index"] << irre_index_;
  ar[group + "KStruct/weight"] << weights_;
  ar[group + "n_star"] << n_star();

  // k, k_conj relations
  ar[group + "KStruct/stars_conj_index"] << stars_conj_index_;
  ar[group + "KStruct/irre_conj_list"] << irre_conj_list_;
  ar[group + "KStruct/irre_conj_index"] << irre_conj_index_;
  ar[group + "KStruct/irre_conj_weight"] << irre_conj_weight_;
  ar[group + "KStruct/conj_list"] << conj_list_;
  ar[group + "KStruct/conj_index"] << conj_index_;
  ar[group + "KStruct/conj_check"] << conj_check_;
  ar[group + "n_conj_star"] << irre_conj_list_.size();
  ar[group + "time_reversal"] << time_reversal();

  for (int i = 0; i < k_mesh_.size(); ++i) {
    ar[group + "Kpoint/" + std::to_string(i) + "/ops"] << ks_ops_[i];
    ar[group + "Kpoint/" + std::to_string(i) + "/little_cogroup"] << little_cogroup_[i];
  }

  // Save information of each star
  std::string prefix = group + "/Star/";

  ar[prefix + "n_star"] << n_star();
  for (int s = 0; s < n_star(); ++s) {
    std::string name = std::to_string(s);

    const auto &star = stars_[s];
    Ndarray<double> star_temp(star.size(), DIM);

    for (int x = 0; x < star.size(); ++x) {
      Ndarray_VecView(star_temp, DIM, x*DIM) = star[x];
      ar[prefix + name + "/operations/k_" + std::to_string(x)] << stars_ops_[s][x];
    }
    ar[prefix + name + "/k_points"] << star_temp;
    ar[prefix + name + "/k_idx"] << stars_index_[s];
  }
}

} // namespace symmetry_adaptation
