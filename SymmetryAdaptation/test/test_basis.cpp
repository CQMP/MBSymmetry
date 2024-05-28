
#include "gtest/gtest.h"
#include "basis.h"

using namespace symmetry_adaptation;

TEST(Basis, InitH) {
  Basis B;
  B.add_atom("H", "1s");
  EXPECT_EQ(B.n_orbitals("H"), 1);
}

TEST(Basis, InitO) {
  Basis B;
  B.add_atom("O", "1s2sp3spd");
  EXPECT_EQ(B.n_orbitals("O"), 14);
  EXPECT_EQ(B.n_orbitals(), 14);
  EXPECT_EQ(B.size(), 1);
}

TEST(Basis, InitNd) {
  Basis B;
  B.add_atom("Nd", "3spd4spdf5s");
  B.add_atom("H", "1s2sp");
  EXPECT_EQ(B.n_orbitals("Nd"), 26);
  EXPECT_EQ(B.n_orbitals("H"), 5);
  EXPECT_EQ(B.n_orbitals(), 31);
  EXPECT_EQ(B.size(), 2);
}

