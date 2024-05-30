#ifndef SYMMETRYMBPT_TIMING_H
#define SYMMETRYMBPT_TIMING_H

/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <mpi.h>
#include <string>
#include <iostream>
#include <vector>
#include <map>

namespace symmetry_mbpt {
  /**
   * @brief ExecutionStatistic class
   *
   * @author iskakoff
   */
  class execution_statistic_t {
  public:

    execution_statistic_t() = default;

    void add(const std::string &name) {
      if (events_.find(name) == events_.end()) {
        events_[name] = std::make_pair(0.0, 0.0);
      }
    }

    /**
     * Update event time
     * @param name - event name
     */
    void end(const std::string &name) {
      double time1 = time();
      events_[name] = std::make_pair(events_[name].first + time1 - events_[name].second, time1);
    }

    /**
     * register the start point of the event
     *
     * @param name - event name
     */
    void start(const std::string &name) {
      events_[name] = std::make_pair(events_[name].first, time());
    }

    /**
     * Print all observed events
     */
    void print() {
      std::cout << "Execution statistics:" << std::endl;
      for (auto &kv : events_) {
        std::cout << "Event " << kv.first << " took " << kv.second.first << "s." << std::endl;
      }
      std::cout << "=====================" << std::endl;
    }

    void print(MPI_Comm comm) {
      int id, np;
      MPI_Comm_rank(comm, &id);
      MPI_Comm_size(comm, &np);

      std::vector<double> max(events_.size(), 0.0);
      size_t i = 0;
      for (auto &kv : events_) {
        max[i] = kv.second.first;
        ++i;
      }
      if (!id) {
        MPI_Reduce(MPI_IN_PLACE, max.data(), max.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
      } else {
        MPI_Reduce(max.data(), max.data(), max.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
      }
      if (!id) {
        std::cout << "Execution statistics: " << std::endl;
        i = 0;
        for (auto &kv : events_) {
          std::cout << "Event " << kv.first << " took " << max[i] << " s." << std::endl;
          ++i;
        }
        std::cout << "=====================" << std::endl;
      }
    }

    /**
     * Return event timing pair
     * @param event_name - event name
     * @return event timing
     */
    std::pair<double, double> event(const std::string &event_name) {
      if (events_.find(event_name) != events_.end()) {
        return events_[event_name];
      }
      return std::make_pair(0.0, 0.0);
    };
  private:
    // registered events timing pairs
    // pair.first corresponds to total event time
    // pair.second corresponds to last time when event was happened
    std::map<std::string, std::pair<double, double> > events_;

    double time() const {
      return MPI_Wtime();
    }
  };

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_TIMING_H
