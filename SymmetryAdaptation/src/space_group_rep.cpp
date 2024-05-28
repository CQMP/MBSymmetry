
#include "space_group_rep.h"

namespace symmetry_adaptation {

namespace space_group_representation {

std::tuple<Matrix<int>, Ndarray<double> >
generate_permutation_matrix(const SpaceGroup &group, const UnitCell &unit_cell, double tol, bool verbose) {

  int n_atom = unit_cell.n_atom();
  int order = group.order();
  Matrix<int> permutation_table(order, n_atom);
  Ndarray<double> shift_table(order, n_atom, DIM);

  ColVector<double, DIM> pos;
  ColVector<double, DIM> pos_target;
  ColVector<double, DIM> trans_pos;
  ColVector<double, DIM> shift_pos;
  for (int g = 0; g < order; ++g) {
    const auto &R = group.space_rep()[g];
    const auto &shift = group.translations()[g];
    for (int i = 0; i < n_atom; ++i) {
      pos = unit_cell.trans_vec().shift_back_to_center_cell(unit_cell.atom(i).pos());
      trans_pos = R * pos + shift;
      shift_pos = unit_cell.trans_vec().shift_back_to_center_cell(trans_pos);

      bool found_partner = false;
      for (int z = 0; (z < n_atom) && !found_partner; ++z) {
        pos_target = unit_cell.trans_vec().shift_back_to_center_cell(unit_cell.atom(z).pos());

        double distance_to_z = (shift_pos - pos_target).norm();
        if (distance_to_z < tol) {
          if (unit_cell.atom(z).element() != unit_cell.atom(i).element())
            throw std::runtime_error("point group maps atoms of different type onto each other");
          found_partner = true;
          permutation_table(g, i) = z;
          Ndarray_VecView(shift_table, DIM, g*n_atom*DIM+i*DIM) = (shift_pos - trans_pos).transpose();
        }
      }

      if (!found_partner) {
        std::cout << "atom position: " << unit_cell.atom(i).pos().transpose() << std::endl;
        std::cout << "symmetry operation: " << g << std::endl;
        std::cout << R << std::endl;
        std::cout << "translation vector: " << shift.transpose() << std::endl;
        std::cout << "transformed position: " << trans_pos.transpose() << std::endl;
        throw std::runtime_error("symmetry analysis could not find partner.");
      }
    }
  }

  if (verbose) {
    std::cout << "shift table" << std::endl;
    std::cout << shift_table << std::endl;
  }

  return std::make_tuple(permutation_table, shift_table);
}

Ndarray<dcomplex> generate_representation(const KSpaceStructure &k_struct, const SpaceGroup &group,
                                          const UnitCell &unit_cell, const WignerD &d_mat,
                                          double tol, bool verbose) {

  const auto &kpts = k_struct.k_mesh().kpts();
  const auto &trans_table = k_struct.transform_table();

  const auto perm = generate_permutation_matrix(group, unit_cell, tol, verbose);
  const auto &permut_table = std::get<0>(perm);
  const auto &shift_table = std::get<1>(perm);

  int nk = k_struct.k_mesh().size();
  int order = group.order();
  int size = unit_cell.n_orbitals();
  int n_atom = unit_cell.n_atom();

  std::cout << "generating representation" << std::endl;

  Ndarray<dcomplex> representation(nk, order, size, size);
  representation.set_zero();

  for (int k = 0; k < nk; ++k) {
    for (int g = 0; g < group.order(); ++g) {
        const auto &trans_k = kpts[trans_table(k, g)];
        auto mat = Ndarray_MatView(representation, size, size, k*group.order()*size*size + g*size*size);

        for (int i = 0; i < n_atom; ++i) {
          const auto &atom = unit_cell.atom(i);
          int target = permut_table(g, i);
          const auto &orb = unit_cell.basis().orbitals(atom.element());
          int n_orb = orb.size();

          auto sv = Ndarray_VecView(shift_table, DIM, g*n_atom*DIM+i*DIM).transpose();

          dcomplex phase = std::exp(std::complex<double>(0, 1.) * trans_k.dot(sv));

          for(int o = 0; o < n_orb; ++o) {
          int n = orb[o].n;
          int l = orb[o].l;
          int stride = 2 * l + 1;
          const auto &d = d_mat.D(l, g);

          int source_start_idx = unit_cell.index_atom_n_l(i, n, l);
          int target_start_idx = unit_cell.index_atom_n_l(target, n, l);
          mat.block(target_start_idx, source_start_idx, stride, stride) = phase * d;
        }
      }
    }
  }

  return representation;
}

} // namespace space_group_representation

void FactorGroupRep::compute_factor_system() {

  const auto &kpts = k_struct_.k_mesh().kpts();

  int nk = k_struct_.k_mesh().size();
  int order = group_.order();

  for (int k = 0; k < nk; ++k) {
    const auto &kpt = kpts[k];
    for (int a = 0; a < order; ++a) {
      const auto &rot = group_.space_rep(a);
      for (int b = 0; b < order; ++b) {
        const auto &trans = group_.translation(b);
        factors_(k, a, b) = std::exp(dcomplex(0, 1.) * kpt.dot(trans - rot * trans));
      }
    }
  }
}

void FactorGroupRep::generate_projective_representation() {

  const auto &kpts = k_struct_.k_mesh().kpts();

  int order = group_.order();
  int nk = k_struct_.k_mesh().size();
  int size = representation_.shape()[3];

  for (int k = 0; k < nk; ++k) {
    const auto &kpt = kpts[k];
    for (int g = 0; g < group_.order(); ++g) {
      dcomplex phase = std::exp(std::complex<double>(0, 1.) * kpt.dot(group_.translations()[g]));
      auto mat = Ndarray_MatView(representation_, size, size, k*order*size*size + g*size*size);
      auto proj_mat = Ndarray_MatView(proj_rep_, size, size, k*order*size*size + g*size*size);
      proj_mat = phase * mat;
    }
  }
}

void FactorGroupRep::check_factors(double tol) const {

  /*
   * Check that the factors are unitary and they satisfy the group multiplication table.
   * lambda(a, bc) * lambda(b, c) = lambda(ab, c) * lambda(a, b)
   */
  int nk = factors_.shape()[0];
  for (int k = 0; k < nk; ++k) {
    const auto &little_cogroup = k_struct_.little_cogroup(k);
    for (int a : little_cogroup) {
      for (int b : little_cogroup) {
        if (std::abs(std::conj(factors_.at(k, a, b)) - 1./factors_.at(k, a, b)) > tol) {
          throw std::runtime_error("Factor system is not unitary.");
        }
        for (int c : little_cogroup) {
          dcomplex left = factors_.at(k, a, group_.multiplication_table()(b, c)) * factors_.at(k, b, c);
          dcomplex right = factors_.at(k, group_.multiplication_table()(a, b), c) * factors_.at(k, a, b);
          if (std::abs(left - right) > tol) {
            throw std::runtime_error("Factor system does not satisfy the group multiplication table.");
          }
        }
      }
    }
  }
}

void FactorGroupRep::check_proj_representation(double tol) const {

  /*
   * Check that the projective representation satisfies the group multiplication table.
   * D(a) * D(b) = lambda(a, b) * D(a * b)
   * D(b) * D(g) = lambda(b, g) * lambda(a, b)^* * D(a) * D(b) with b*g = a*b
   */
  int order = group_.order();
  int nk = factors_.shape()[0];
  int size = proj_rep_.shape()[3];
  Matrix<dcomplex> left(size, size);
  Matrix<dcomplex> right(size, size);
  for (int k = 0; k < nk; ++k) {
    const auto &little_cogroup = k_struct_.little_cogroup(k);
    for (int a : little_cogroup) {
      for (int b : little_cogroup) {
        int c = group_.multiplication_table()(a, b);
        left = Ndarray_MatView(proj_rep_, size, size, k*order*size*size + a*size*size)
               * Ndarray_MatView(proj_rep_, size, size, k*order*size*size + b*size*size);
        right = factors_.at(k, a, b) * Ndarray_MatView(proj_rep_, size, size, k*order*size*size + c*size*size);
        if ((left - right).norm() > tol) {
          throw std::runtime_error("Projective representations do not satisfy the group multiplication table.");
        }
        for (int g: little_cogroup) {
          if (group_.multiplication_table()(b, g) == c) {
            left = Ndarray_MatView(proj_rep_, size, size, k*order*size*size + b*size*size)
                   * Ndarray_MatView(proj_rep_, size, size, k*order*size*size + g*size*size);
            right = factors_.at(k, b, g) * std::conj(factors_.at(k, a, b))
              * Ndarray_MatView(proj_rep_, size, size, k*order*size*size + a*size*size)
              * Ndarray_MatView(proj_rep_, size, size, k*order*size*size + b*size*size);
            if ((left - right).norm() > tol) {
              throw std::runtime_error("Projective representations do not satisfy the group multiplication table.");
            }
          }
        }
      }
    }
  }
}

std::vector<std::vector<std::vector<Matrix<dcomplex> > > > FactorGroupRep::check_conjugate_relation() const {
  // Check equation B9 in Dovesi's paper

  int nk = k_struct_.k_mesh().size();
  int order = group_.order();
  int size = representation_.shape()[3];
  Matrix<dcomplex> Omega_c(size, size);
  std::vector<std::vector<std::vector<Matrix<dcomplex> > > > conjugacy_class_sum; //(k, i, g)
  conjugacy_class_sum.resize(nk);
  for (int k = 0; k < nk; ++k) {
    const auto &little_cogroup = k_struct_.little_cogroup(k);
    const auto &conjugacy_classes = k_struct_.little_cogroup_conjugacy_classes_mult(k);
    int n_cc = conjugacy_classes.size();
    conjugacy_class_sum[k].resize(n_cc);
    for (int i = 0; i < n_cc; ++i) {
      const auto &conjugacy_class = conjugacy_classes[i];
      for (int g: conjugacy_class) {
        Omega_c.setZero();
        for (int b: little_cogroup) {
          bool found = false;
          for (int a : conjugacy_class) {
            if (group_.multiplication_table()(a, b) == group_.multiplication_table()(b, g)) {
              Omega_c += factors_.at(k, b, g) * std::conj(factors_.at(k, a, b))
                * Ndarray_MatView(proj_rep_, size, size, k*order*size*size + a*size*size);
              found = true;
              break;
            }
          } // a
          if (!found) {
            std::cout << g << " " << b << std::endl;
            for (int ccel :conjugacy_class) std::cout << ccel << " ";
            std::cout << std::endl;
            throw std::runtime_error("can not find conjugate operation for k point " + std::to_string(k));
          }
        } // b
        conjugacy_class_sum[k][i].push_back(Omega_c * conjugacy_class.size() / little_cogroup.size());
      } // g

      // check all matrices in conjugacy_class_sum[k][i] are differ only by a unitary phase
      for (int ig1 = 0; ig1 < conjugacy_class.size(); ++ig1) {
        int g1 = conjugacy_class[ig1];
        for (int ig2 = 0; ig2 < conjugacy_class.size(); ++ig2) {
          int g2 = conjugacy_class[ig2];
          for (int b: little_cogroup) {
            if (group_.multiplication_table()(b, g1) == group_.multiplication_table()(g2, b)) {
              dcomplex phase = std::conj(factors_.at(k, b, g1)) * factors_.at(k, g2, b);
              // reuse Omega_c
              Omega_c = conjugacy_class_sum[k][i][ig2] - phase * conjugacy_class_sum[k][i][ig1];
              if (Omega_c.norm() > tol_) {
                std::cout << "phase: " << phase << std::endl;
                std::cout << conjugacy_class_sum[k][i][ig2].norm() << std::endl;
                std::cout << conjugacy_class_sum[k][i][ig1].norm() << std::endl;
                throw std::runtime_error(
                  "Omega_c computed using different operations in a class does not differ only in a unitary phase.");
              }
            }
          } // b
        } // g2
      } // g1

    } // i
  } // k
  return conjugacy_class_sum;
}

} // namespace symmetry_adaptation
