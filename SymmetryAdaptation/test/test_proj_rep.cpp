
#include "gtest/gtest.h"
#include "diagonalization.h"
#include "space_group_rep.h"
#include "basis.h"
#include "translation_vector.h"
#include "unit_cell.h"

using namespace symmetry_adaptation;

TEST(ProjRep, Si) {
  bool verbose = false;

  std::stringstream xyz;
  xyz << "2\nSi\n";
  xyz << "Si 0.     0.     0.  \n";
  xyz << "Si 1.3575 1.3575 1.3575\n";

  std::stringstream a;
  a << "2.715  2.715  0.0\n";
  a << "0.0  2.715  2.715\n";
  a << "2.715  0.0  2.715\n";

  Basis basis;
  basis.add_atom("Si", "1s2sp");
  basis.set_atom_types();

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  int q = 4;
  KMesh k_mesh(q, tvec);
  k_mesh.generate_k_mesh("pyscf");

  double tol = 1e-6;

  SpaceGroup group_space(tol);
  EXPECT_NO_THROW(group_space.get_space_group_info(unit_cell, true));
  EXPECT_EQ(group_space.symmorphic(), false);

  WignerD wigner_d_space(0, "pyscf");
  wigner_d_space.compute(group_space);
  std::unique_ptr<KSpaceStructure> k_struct_space
    = std::make_unique<KSpaceStructure>(k_mesh, group_space, tvec, tol, verbose, true, true);
  Ndarray<dcomplex> orbital_rep_space;
  EXPECT_NO_THROW(orbital_rep_space =
    space_group_representation::generate_representation(*k_struct_space, group_space,
                                                        unit_cell, wigner_d_space, tol, verbose));

  FactorGroupRep factor_group_rep(*k_struct_space, group_space, orbital_rep_space, 1e-6);
  EXPECT_NO_THROW(factor_group_rep.check_multiplication());
  EXPECT_NO_THROW(factor_group_rep.check_conjugate_relation());

  alps::hdf5::archive output_file("temp.h5", "w");
  EXPECT_NO_THROW(diagonalization::diagonalize_proj_dirac_characters(k_struct_space, group_space, orbital_rep_space,
                                                                     output_file, true, 1e-6, true));

  // include time reversal symmetry
  EXPECT_NO_THROW(diagonalization::diagonalize_proj_dirac_characters(k_struct_space, group_space, orbital_rep_space,
                                                                     output_file, true, 1e-6, true, true));
  output_file.close();
}

TEST(ProjRep, LiH) {
  bool verbose = false;

  std::stringstream xyz;
  xyz << "2\nLiH\n";
  xyz << "H 0.     0.     0.  \n";
  xyz << "Li 2.03275 2.03275 2.03275\n";

  std::stringstream a;
  a << "2.03275  2.03275  0.0\n";
  a << "0.0  2.03275  2.03275\n";
  a << "2.03275  0.0  2.03275\n";

  Basis basis;
  basis.add_atom("H", "1s2sp");
  basis.add_atom("Li", "1s2sp3s");
  basis.set_atom_types();

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  double tol = 1e-8;

  SpaceGroup group_space(tol);
  EXPECT_NO_THROW(group_space.get_space_group_info(unit_cell, true));
  std::cout << group_space.space_group_number() << std::endl;
  EXPECT_EQ(group_space.symmorphic(), true);

  WignerD wigner_d_space(0, "pyscf");
  wigner_d_space.compute(group_space);

  int q = 2;
  KMesh k_mesh(q, tvec);
  k_mesh.generate_k_mesh("pyscf");
  std::unique_ptr<KSpaceStructure> k_struct_space
    = std::make_unique<KSpaceStructure>(k_mesh, group_space, tvec, tol, verbose, true, true);
  Ndarray<dcomplex> orbital_rep_space;
  EXPECT_NO_THROW(orbital_rep_space =
                    space_group_representation::generate_representation(*k_struct_space, group_space,
                                                                        unit_cell, wigner_d_space, tol, verbose));

  FactorGroupRep factor_group_rep(*k_struct_space, group_space, orbital_rep_space, tol);
  EXPECT_NO_THROW(factor_group_rep.check_multiplication());
  EXPECT_NO_THROW(factor_group_rep.check_conjugate_relation());

  alps::hdf5::archive output_file("temp.h5", "w");
  EXPECT_NO_THROW(diagonalization::diagonalize_proj_dirac_characters(k_struct_space, group_space, orbital_rep_space,
                                                                     output_file, true, tol, true));

  // include time reversal symmetry
  EXPECT_NO_THROW(diagonalization::diagonalize_proj_dirac_characters(k_struct_space, group_space, orbital_rep_space,
                                                                     output_file, true, tol, true, true));
  output_file.close();
}
