
#include "gtest/gtest.h"
#include "atom.h"
#include "type.h"

using namespace symmetry_adaptation;

TEST(Atom, Init) {
  ColVector<double, 3> pos = {0., 0., 0.};
  Atom H("H", pos);
  EXPECT_EQ(H.element(), "H");
  EXPECT_EQ(H.pos(), pos);
}

TEST(Atom, Oxygen) {
  using namespace symmetry_adaptation;

  ColVector<double, 3> pos = {1., 2., 3.};
  Atom O("O", pos);
  EXPECT_EQ(O.element(), "O");
  EXPECT_EQ(O.pos(), pos);
}

