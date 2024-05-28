
#include "diagonalization.h"
#include "space_group_rep.h"
#include "numeric.h"

namespace symmetry_adaptation {

namespace diagonalization {

void diagonalize_dirac_characters(const std::unique_ptr<KSpaceStructure> &k_struct,
                                  Ndarray<dcomplex> &orbital_rep,
                                  alps::hdf5::archive &output_file,
                                  bool use_lapack, double tol, bool verbose,
                                  bool time_reversal) {

  int orep_size = orbital_rep.shape()[orbital_rep.dim()-1];
  int nk = k_struct->k_mesh().size();
  const auto &star_rep_index = k_struct->star_rep_index();
  for (int i = 0; i < nk; ++i) {
    if (verbose) {
      std::cout << "analysing Dirac character of k point " << i << std::endl;
    }
    DiracCharacter dirac(k_struct->little_cogroup_conjugacy_classes(i).size(), orep_size);
    dirac.calculate_conjugacy_class_sum(k_struct->little_cogroup_conjugacy_classes(i), orbital_rep(i));
    dirac.U() = SimulDiag::solve(dirac.conjugacy_class_sum(), tol, use_lapack);
    dirac.compute_diagonal_elements();
    dirac.find_block_size(tol);

    // ith k point is conjugate with itself
    if (k_struct->k_mesh().self_conjugate()[i] && time_reversal) {
      std::cout << "applying time reversal relation to k point " << i << std::endl;
      apply_time_reversal_relation(dirac, tol);
    }

    std::string prefix = "/Kpoint/" + std::to_string(i);
    dirac.save(output_file, prefix);

    auto it = std::find(star_rep_index.begin(), star_rep_index.end(), i);
    if (it != star_rep_index.end()) {
      int star = it - star_rep_index.begin();
      prefix = "/Star/" + std::to_string(star);
      dirac.save(output_file, prefix);
    }
  }
}

void diagonalize_proj_dirac_characters(const std::unique_ptr<KSpaceStructure> &k_struct,
                                       const SpaceGroup &group,
                                       Ndarray<dcomplex> &orbital_rep,
                                       alps::hdf5::archive &output_file,
                                       bool use_lapack, double tol, bool verbose,
                                       bool time_reversal) {
  if (group.symmorphic()) {
    diagonalize_dirac_characters(k_struct, orbital_rep, output_file, use_lapack, tol, verbose, time_reversal);
    return;
  }

  // compute projected representation and factors
  FactorGroupRep factor_group_rep((*k_struct), group, orbital_rep, tol);
  // temporarily put it here just to be safe
  factor_group_rep.check_multiplication();

  auto proj_rep = factor_group_rep.proj_rep();
  auto factors = factor_group_rep.factors();

  int orep_size = orbital_rep.shape()[orbital_rep.dim()-1];
  int nk = k_struct->k_mesh().size();
  const auto &star_rep_index = k_struct->star_rep_index();
  for (int i = 0; i < nk; ++i) {
    if (verbose) {
      std::cout << "analysing Dirac character of k point " << i << std::endl;
    }
    DiracCharacter dirac(k_struct->little_cogroup_conjugacy_classes(i).size(), orep_size);
    dirac.calculate_proj_conjugacy_class_sum(group, k_struct->little_cogroup(i),
                                             k_struct->little_cogroup_conjugacy_classes(i),
                                             proj_rep(i), factors(i), tol);

    dirac.U() = SimulDiag::solve(dirac.conjugacy_class_sum(), tol, use_lapack);
    dirac.compute_diagonal_elements();
    dirac.find_block_size(tol);

    // ith k point is conjugate with itself
    if (k_struct->k_mesh().self_conjugate()[i] && time_reversal) {
      std::cout << "applying time reversal relation to k point " << i << std::endl;
      apply_time_reversal_relation(dirac, tol);
    }

    std::string prefix = "/Kpoint/" + std::to_string(i);
    dirac.save(output_file, prefix);

    auto it = std::find(star_rep_index.begin(), star_rep_index.end(), i);
    if (it != star_rep_index.end()) {
      int star = it - star_rep_index.begin();
      prefix = "/Star/" + std::to_string(star);
      dirac.save(output_file, prefix);
    }
  }
}

void apply_time_reversal_relation(DiracCharacter &dirac, double tol) {

  // Check whether the current U matrix fulfills the relation in Dovesi's paper
  int n_irreps = dirac.block_start_idx().size();
  auto block_idx = dirac.block_start_idx();
  block_idx.push_back(dirac.size());
  ColVector<dcomplex> Ui_conj;
  for (int n = 0; n < n_irreps; ++n) {
    int start = block_idx[n];
    int end = block_idx[n+1];
    const auto U_irrep = dirac.U().block(0, start, dirac.U().rows(), end - start);
    bool in_span = true;
    for (int i = start; i < end; ++i) {
      const auto &Ui = dirac.U().col(i);
      Ui_conj = Ui.conjugate();
      // Check whether Ui_conj is in the span of U[:, st:en]
      in_span *= numeric::is_in_span(Ui_conj, U_irrep, tol);
    }
    if (!in_span) {
      throw std::logic_error("U conj is not in the span of U");
    }
  }

  // Find real eigenvectors and update the U matrix
  for (int n = 0; n < n_irreps; ++n) {
    int start = block_idx[n];
    int end = block_idx[n+1];
    std::vector<ColVector<double> > U_list;
    for (int i = start; i < end; ++i) {
      const auto &Ui = dirac.U().col(i);
      if (Ui.real().cwiseAbs().maxCoeff() > tol) {
        U_list.push_back(Ui.real());
      }
      if (Ui.imag().cwiseAbs().maxCoeff() > tol) {
        U_list.push_back(Ui.imag());
      }
    }
    int ite = 0;
    while (U_list.size() > end-start && ite < 3) {
      numeric::get_orthonormal_basis_householder(U_list, tol);
      ite += 1;
    }
    if (U_list.size() != end-start) {
      std::cout << end - start << std::endl;
      std::cout << U_list.size() << std::endl;
      throw std::runtime_error("fail to find orthonormal time reversal basis");
    }
    for (int i = start; i < end; ++i) {
      dirac.U().col(i) = U_list[i-start];
    }
  }
  SimulDiag::check_diag(dirac.conjugacy_class_sum(), dirac.U(), tol);

  dirac.compute_diagonal_elements();
  dirac.find_block_size(tol);
}

} // namespace diagonalization

} // namespace symmetry_adaptation
