#ifndef SYMMETRYMBPT_MPI_UTILS_H
#define SYMMETRYMBPT_MPI_UTILS_H

/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <stdexcept>
#include <iostream>
#include <complex>
#include <string>
#include <mpi.h>

template<typename prec>
struct mpi_type {
};
template<>
struct mpi_type<double> {
  static MPI_Datatype complex_type;
  static MPI_Datatype scalar_type;
};
template<>
struct mpi_type<float> {
  static MPI_Datatype complex_type;
  static MPI_Datatype scalar_type;
};

void setup_intranode_communicator(MPI_Comm &global_comm, const int &global_rank,
                                  MPI_Comm &intranode_comm, int &intranode_rank, int &intranode_size);

void setup_devices_communicator(MPI_Comm &global_comm, const int &global_rank,
                                const int &intranode_rank, const int &devCount_per_node, const int &devCount_total,
                                MPI_Comm &devices_comm, int &devices_rank, int &devices_size);

void setup_internode_communicator(MPI_Comm &global_comm, const int &global_rank,
                                  const int &intranode_rank,
                                  MPI_Comm &internode_comm, int &internode_rank, int &internode_size);

template<typename T>
void setup_mpi_shared_memory(T **ptr_to_shared_mem, MPI_Aint &buffer_size, MPI_Win &shared_win,
                             MPI_Comm &intranode_comm, int intranode_rank) {
  int disp_unit;
  // Allocate shared memory buffer (i.e. shared_win) on local process 0 of each shared-memory communicator (i.e. of each node)
  if (MPI_Win_allocate_shared((!intranode_rank) ? buffer_size : 0, sizeof(T),
                              MPI_INFO_NULL, intranode_comm, ptr_to_shared_mem, &shared_win) != MPI_SUCCESS)
    throw std::runtime_error("Failed allocating shared memory.");

  // This will be called by all processes to query the pointer to the shared area on local zero process.
  if (MPI_Win_shared_query(shared_win, 0, &buffer_size, &disp_unit, ptr_to_shared_mem) != MPI_SUCCESS)
    throw std::runtime_error("Failed extracting pointer to the shared area)");
  MPI_Barrier(intranode_comm);
}

#endif //SYMMETRYMBPT_MPI_UTILS_H
