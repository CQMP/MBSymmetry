
#include "gtest/gtest.h"
#include "space_group.h"
#include "basis.h"
#include "unit_cell.h"
#include "translation_vector.h"

using namespace symmetry_adaptation;

TEST(SpaceGroup, SrVO3) {

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
  basis.set_atom_types();

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  double tol = 1e-6;

  SpaceGroup group_space(tol);
  EXPECT_NO_THROW(group_space.get_space_group_info(unit_cell, true));
  EXPECT_EQ(group_space.symmorphic(), true);
}

TEST(SpaceGroup, Si) {
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
  basis.add_atom("Si", "1s2sp3sp");
  basis.set_atom_types();

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  double tol = 1e-6;

  SpaceGroup group_space(tol);
  EXPECT_NO_THROW(group_space.get_space_group_info(unit_cell, true));
  EXPECT_EQ(group_space.symmorphic(), false);
}
