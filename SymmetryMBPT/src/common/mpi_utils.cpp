/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <stdexcept>
#include "mpi_utils.h"

MPI_Datatype mpi_type<double>::complex_type = MPI_C_DOUBLE_COMPLEX;
MPI_Datatype mpi_type<double>::scalar_type = MPI_DOUBLE;
MPI_Datatype mpi_type<float>::complex_type = MPI_C_FLOAT_COMPLEX;
MPI_Datatype mpi_type<float>::scalar_type = MPI_FLOAT;

void setup_intranode_communicator(MPI_Comm &global_comm, const int &global_rank,
                                  MPI_Comm &intranode_comm, int &intranode_rank, int &intranode_size) {
  // Split world_comm_ into several sub-communicators, one per shared memory domain,
  // In addition, one shared memory domain is equivalent to one node.
  if ((MPI_Comm_split_type(global_comm, MPI_COMM_TYPE_SHARED, global_rank, MPI_INFO_NULL, &intranode_comm)) !=
      MPI_SUCCESS)
    throw std::runtime_error("Failed splitting shared-memory communicators.");
  MPI_Comm_rank(intranode_comm, &intranode_rank);
  MPI_Comm_size(intranode_comm, &intranode_size);
}

void setup_devices_communicator(MPI_Comm &global_comm, const int &global_rank,
                                const int &intranode_rank, const int &devCount_per_node, const int &devCount_total,
                                MPI_Comm &devices_comm, int &devices_rank, int &devices_size) {
  if (intranode_rank < devCount_per_node) {
    int color = 0;
    MPI_Comm_split(global_comm, color, global_rank, &devices_comm);
    MPI_Comm_rank(devices_comm, &devices_rank);
    MPI_Comm_size(devices_comm, &devices_size);
    if (devices_size != devCount_total)
      throw std::runtime_error("Number of devices mismatches size of devices' communicator.");
  } else {
    MPI_Comm_split(global_comm, MPI_UNDEFINED, global_rank, &devices_comm);
    devices_rank = -1;
    devices_size = -1;
  }
}

void setup_internode_communicator(MPI_Comm &global_comm, const int &global_rank,
                                  const int &intranode_rank,
                                  MPI_Comm &internode_comm, int &internode_rank, int &internode_size) {
  if (!intranode_rank) {
    MPI_Comm_split(global_comm, intranode_rank, global_rank, &internode_comm);
    MPI_Comm_rank(internode_comm, &internode_rank);
    MPI_Comm_size(internode_comm, &internode_size);
    if (!global_rank && internode_rank != global_rank) throw std::runtime_error("Root rank mismatched!");
  } else {
    MPI_Comm_split(global_comm, MPI_UNDEFINED, global_rank, &internode_comm);
    internode_rank = -1;
    internode_size = -1;
  }
}
