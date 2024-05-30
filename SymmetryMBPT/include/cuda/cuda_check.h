#ifndef SYMMETRYMBPT_CUDA_CHECK_H
#define SYMMETRYMBPT_CUDA_CHECK_H

#include <mpi.h>

void check_for_cuda_mpi(MPI_Comm global_comm, int global_rank, int &devCount_per_node);

#endif //SYMMETRYMBPT_CUDA_CHECK_H
