
#include "gtest/gtest.h"

#include "spglib.h"
#include "spglib_anlys.h"
#include "type.h"

using namespace symmetry_adaptation;

TEST(SpgLib, WurtziteStructure) {
  SpglibDataset *dataset;

  Matrix<double, DIM, DIM> a_vectors;
  a_vectors << 3.111, -1.5555, 0, 0, 2.6942050311733885, 0, 0, 0, 4.988;

  double lattice[3][3];
  MatrixMap<double, DIM, DIM>(&lattice[0][0], DIM, DIM) = a_vectors;

  CMatrix<double> pos(DIM, 4);
  pos << 1.0 / 3, 2.0 / 3, 0.0,
         2.0 / 3, 1.0 / 3, 0.5,
         1.0 / 3, 2.0 / 3, 0.6181,
         2.0 / 3, 1.0 / 3, 0.1181;

  double position[4][DIM];
  CMatrixMap<double>(&position[0][0], 4, DIM) = pos.transpose();

  std::vector<int> types{1, 1, 2, 2};

  int num_atom = 4;
  double symprec = 1e-5;

  dataset = spg_get_dataset(lattice, position, types.data(), num_atom, symprec);

  EXPECT_EQ(dataset->spacegroup_number, 186);

  spg_free_dataset(dataset);
}

TEST(SpgLibAnlys, SrVO3) {

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
  basis.atom_types()["Sr"] = 0;
  basis.atom_types()["V"] = 1;
  basis.atom_types()["O"] = 2;

  TranslationVector tvec(a.str());
  UnitCell unit_cell(tvec, xyz.str(), basis, verbose);

  SpglibDataset *dataset;
  EXPECT_NO_THROW(dataset = spglib_anlys::get_symmetry_info(unit_cell, 1e-9));

  EXPECT_EQ(dataset->n_atoms, 5);
  EXPECT_EQ(dataset->spacegroup_number, 221);
  EXPECT_EQ(dataset->n_operations, 48);
}

