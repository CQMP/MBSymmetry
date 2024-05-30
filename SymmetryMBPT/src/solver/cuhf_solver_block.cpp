
#include "analysis.h"
#include "cuhf_solver_block.h"

namespace symmetry_mbpt {

  void cuhf_solver_block_t::setup_MPI_structure() {
    devCount_total_ = (intranode_rank_ < devCount_per_node_) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &devCount_total_, 1, MPI_INT, MPI_SUM, world_comm_);
    if (!world_rank_)
      std::cout << "Your host has " << devCount_per_node_ << " devices/node and we'll use " << devCount_total_
                << " devices in total." << std::endl;
    setup_devices_communicator(world_comm_, world_rank_, intranode_rank_,
                               devCount_per_node_, devCount_total_, devices_comm_,
                               devices_rank_, devices_size_);
  }

  void cuhf_solver_block_t::clean_MPI_structure() {
    if (devices_comm_ != MPI_COMM_NULL) {
      if (MPI_Comm_free(&devices_comm_) != MPI_SUCCESS) throw std::runtime_error("Fail releasing device communicator");
    }
  }

} // namespace symmetry_mbpt