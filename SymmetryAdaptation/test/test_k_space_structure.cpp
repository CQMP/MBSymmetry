
#include "gtest/gtest.h"
#include "diagonalization.h"
#include "basis.h"
#include "translation_vector.h"
#include "unit_cell.h"

using namespace symmetry_adaptation;

TEST(KSpaceStructure, Si) {
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

  EXPECT_NO_THROW(std::unique_ptr<KSpaceStructure> k_struct_space
    = std::make_unique<KSpaceStructure>(k_mesh, group_space, tvec, tol, verbose, true, true));
}

TEST(KSpaceStructure, LiH) {

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
  basis.add_atom("Li", "1s2sp");
  basis.add_atom("H", "1s");
  basis.set_atom_types();

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  int q = 3;
  KMesh k_mesh(q, tvec);
  k_mesh.generate_k_mesh("pyscf");

  double tol = 1e-6;
  SpaceGroup group_space(tol);
  group_space.get_space_group_info(unit_cell, true);

  EXPECT_NO_THROW(std::unique_ptr<KSpaceStructure> k_struct_space
    = std::make_unique<KSpaceStructure>(k_mesh, group_space, tvec, tol, verbose, true, true));
}

