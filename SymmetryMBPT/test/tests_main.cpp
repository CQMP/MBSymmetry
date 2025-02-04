
#include <mpi.h>
#include "gtest/gtest.h"

int main(int argc, char** argv) {
  int result = 0;
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
