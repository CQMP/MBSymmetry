
#include "gtest/gtest.h"
#include "basis.h"
#include "molecule.h"

using namespace symmetry_adaptation;

TEST(Molecule, H2) {

  std::stringstream xyz;
  xyz << "2\nHydrogen\n";
  xyz << "H 0 0 0\n";
  xyz << "H 0 0 0.74\n";

  EXPECT_NO_THROW(Molecule M(xyz.str(), false));
}

TEST(Molecule, NH3) {

  std::stringstream xyz;
  xyz << "4\nAmmonia\n";
  xyz << "N 0 0 0.4\n";
  xyz << "H 0.6 0.6 0\n";
  xyz << "H 0.219615 -0.819615 0\n";
  xyz << "H -0.819615 0.219615 0\n";

  EXPECT_NO_THROW(Molecule M(xyz.str(), false));
}

TEST(Molecule, CH4) {

  std::stringstream xyz;
  xyz << "5\nMethane\n";
  xyz << "H      0.5288      0.1610      0.9359\n";
  xyz << "C      0.0000      0.0000      0.0000\n";
  xyz << "H      0.2051      0.8240     -0.6786\n";
  xyz << "H      0.3345     -0.9314     -0.4496\n";
  xyz << "H     -1.0685     -0.0537      0.1921\n";

  EXPECT_NO_THROW(Molecule M(xyz.str(), false));
}

TEST(Molecule, Init) {
  std::stringstream xyz;
  xyz << "4\nAmmonia\n";
  xyz << "N 0 0 0.4\n";
  xyz << "H 0.848528 0. 0.\n";
  xyz << "H -0.424264 -0.734847 0\n";
  xyz << "H -0.424264 0.734847 0\n";

  Molecule M(xyz.str(), false);
  EXPECT_NO_THROW(M.basis().add_atom("H", "1s"));
  EXPECT_NO_THROW(M.basis().add_atom("N", "1s2sp"));

  Basis B;
  B.add_atom("H", "1s");
  B.add_atom("N", "1s2sp");
  EXPECT_NO_THROW(Molecule M1(xyz.str(), B, false));
}

TEST(Molecule, AtomMissingInBasis) {
  std::stringstream xyz;
  xyz<<"4\nAmmonia\n";
  xyz<<"N 0 0 0.4\n";
  xyz<<"H 0.848528 0. 0.\n";
  xyz<<"H -0.424264 -0.734847 0\n";
  xyz<<"H -0.424264 0.734847 0\n";

  Basis B;
  B.add_atom("H", "1s");

  EXPECT_THROW(Molecule M(xyz.str(), B, false), std::runtime_error);
}
