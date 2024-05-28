
#include "gtest/gtest.h"
#include "utils.h"

TEST(Utils, MatrixDumpLoad) {

  using namespace symmetry_adaptation;

  int n = 5;
  Matrix<double> M = Matrix<double>::Random(n, n);

  std::string h5file = "temp.h5";
  alps::hdf5::archive ar_out(h5file, "w");
  ar_out["matrix"] << M;
  ar_out.close();

  Matrix<double> N;
  alps::hdf5::archive ar_in(h5file, "r");
  ar_in["matrix"] >> N;
  ar_in.close();

  EXPECT_LE((M-N).maxCoeff(), 1e-12);
}

TEST(Utils, MatrixDumpLoadComplex) {

  using namespace symmetry_adaptation;

  int n = 5;
  Matrix<dcomplex> M = Matrix<dcomplex>::Random(n, n);

  std::string h5file = "temp.h5";
  alps::hdf5::archive ar_out(h5file, "w");
  ar_out["matrix_complex"] << M;
  ar_out.close();

  Matrix<dcomplex> N;
  alps::hdf5::archive ar_in(h5file, "r");
  ar_in["matrix_complex"] >> N;
  ar_in.close();

  EXPECT_LE((M-N).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(Utils, NdarrayDumpLoad) {

  using namespace symmetry_adaptation;

  Ndarray<double> A(1, 2, 3);

  Ndarray_MatView(A, 2, 3) = Matrix<double>::Random(2, 3);

  std::string h5file = "temp.h5";
  alps::hdf5::archive ar_out(h5file, "w");
  ar_out["ndarray"] << A;
  ar_out.close();

  Ndarray<double> B(1, 2, 3);
  alps::hdf5::archive ar_in(h5file, "r");
  ar_in["ndarray"] >> B;
  ar_in.close();

  ASSERT_TRUE(std::equal(A.begin(), A.end(), B.begin(),
                         [](double a, double b) {return std::abs(a-b)<1e-12;})
  );
}

TEST(Utils, NdarrayDumpLoadComplex) {

  using namespace symmetry_adaptation;

  Ndarray<dcomplex> A(1, 2, 3);

  Ndarray_MatView(A, 2, 3) = Matrix<dcomplex>::Random(2, 3);

  std::string h5file = "temp.h5";
  alps::hdf5::archive ar_out(h5file, "w");
  ar_out["ndarray_complex"] << A;
  ar_out.close();

  Ndarray<dcomplex> B(1, 2, 3);
  alps::hdf5::archive ar_in(h5file, "r");
  ar_in["ndarray_complex"] >> B;
  ar_in.close();

  ASSERT_TRUE(std::equal(A.begin(), A.end(), B.begin(),
                         [](const dcomplex& a, const dcomplex& b) {return std::abs(a-b)<1e-12;})
  );
}

TEST(Utils, MatrixDumpLoadComplexFail) {

  using namespace symmetry_adaptation;

  int n = 5;
  Ndarray<double> M(n, n, n);
  for (int i = 0; i < n; ++i) {
    Ndarray_MatView(M, n, n, i*n*n) = Matrix<double>::Random(n, n);
  }

  std::string h5file = "temp.h5";
  alps::hdf5::archive ar_out(h5file, "w");
  ar_out["ndarray_matrix"] << M;
  ar_out.close();

  Matrix<dcomplex> N;
  alps::hdf5::archive ar_in(h5file, "r");
  EXPECT_THROW(ar_in["ndarray_matrix"] >> N, std::runtime_error);
  ar_in.close();
}
