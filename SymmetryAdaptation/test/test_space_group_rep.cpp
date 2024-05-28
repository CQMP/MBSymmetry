
#include "gtest/gtest.h"
#include "space_group_rep.h"

using namespace symmetry_adaptation;

TEST(SpaceGroupRep, SrVO3) {

  bool verbose = false;

  std::stringstream xyz;
  xyz << "5\nSrVO3\n";
  xyz << "Sr 0.0  0.0  0.0\n";
  xyz << "V 1.9205 1.9205 1.9205\n";
  xyz << "O 1.9205 1.9205 0.0\n";
  xyz << "O 1.9205 0.0 1.9205\n";
  xyz << "O 0.0 1.9205 1.9205\n";

  std::stringstream a;
  a << "3.841  0.0  0.0\n";
  a << "0.0  3.841  0.0\n";
  a << "0.0  0.0  3.841\n";

  Basis basis;
  basis.add_atom("Sr", "1s2sp3sp");  // this is the sto-3g basis of Mg
  basis.add_atom("V", "1s2sp3spd4sp");
  basis.add_atom("O", "1s2sp");

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  int q = 2;
  KMesh k_mesh(q, tvec);
  k_mesh.generate_k_mesh("pyscf");

  double tol = 1e-6;

  SpaceGroup group_space(tol);
  group_space.get_space_group_info(unit_cell);
  EXPECT_EQ(group_space.space_group_number(), 221);

  WignerD wigner_d_space(0, "pyscf");
  wigner_d_space.compute(group_space);
  KSpaceStructure k_struct_space(k_mesh, group_space, tvec, tol, verbose);
  EXPECT_NO_THROW(Ndarray<dcomplex> orbital_rep_space =
    space_group_representation::generate_representation(k_struct_space, group_space,
                                                        unit_cell, wigner_d_space, tol, verbose));
}

TEST(SpaceGroupRep, Si) {

  bool verbose = false;

  std::stringstream xyz;
  xyz << "8\nSi\n";
  xyz << "Si 4.08 4.08 1.36\n";
  xyz << "Si 1.36 4.08 4.08\n";
  xyz << "Si 4.08 1.36 4.08\n";
  xyz << "Si 1.36 1.36 1.36\n";
  xyz << "Si 2.72 0.   2.72\n";
  xyz << "Si 2.72 2.72 0.  \n";
  xyz << "Si 0.   0.   0.  \n";
  xyz << "Si 0.   2.72 2.72\n";

  std::stringstream a;
  a << "5.44  0.0  0.0\n";
  a << "0.0  5.44  0.0\n";
  a << "0.0  0.0  5.44\n";

  Basis basis;
  basis.add_atom("Si", "1s2sp3sp");
  basis.set_atom_types();

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  int q = 2;
  KMesh k_mesh(q, tvec);
  k_mesh.generate_k_mesh("pyscf");

  double tol = 1e-6;

  SpaceGroup group_space(tol);
  group_space.get_space_group_info(unit_cell, false);
  EXPECT_EQ(group_space.space_group_number(), 227);

  WignerD wigner_d_space(0, "pyscf");
  wigner_d_space.compute(group_space);
  KSpaceStructure k_struct_space(k_mesh, group_space, tvec, tol, verbose, false);
  Ndarray<dcomplex> orbital_rep_space;

  EXPECT_NO_THROW(orbital_rep_space=
    space_group_representation::generate_representation(k_struct_space, group_space,
                                                        unit_cell, wigner_d_space, tol, verbose));
}