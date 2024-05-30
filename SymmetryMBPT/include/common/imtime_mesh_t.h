#ifndef SYMMETRYMBPT_IMTIME_MESH_T_H
#define SYMMETRYMBPT_IMTIME_MESH_T_H

#include <cmath>
#include <alps/hdf5.hpp>

namespace symmetry_mbpt {
/**
 * @brief Imaginary-time mesh class defined at the Chebyshev nodes with additional boundary points
 */
class itime_mesh_t {
  double beta_;
  size_t ntau_;
  std::vector<double> points_;
public:

  itime_mesh_t() : beta_(0.0), ntau_(0) {}

  itime_mesh_t(const itime_mesh_t &rhs) : beta_(rhs.beta_), ntau_(rhs.ntau_) { compute_points_cheby(); }

  itime_mesh_t(size_t ntau, double beta, int eta = 1, bool IR = false,
               const std::string &path = "IR_path") : beta_(beta), ntau_(ntau - 1 + eta) {
    if (!IR) {
      compute_points_cheby();
    } else {
      compute_points_ir(path, eta);
    }
  }

  double operator[](size_t idx) const {
    return points_[idx];
  }

  //Getter variables for members
  /// number of tau-points
  size_t extent() const { return ntau_; }

  /// inverse temperature
  double beta() const { return beta_; }

  /// vector of points
  const std::vector<double> &points() const { return points_; }

  /// Compare for equality
  bool operator==(const itime_mesh_t &mesh) const {
    return beta_ == mesh.beta_ && ntau_ == mesh.ntau_;
  }

  /// Compare for non-equality
  bool operator!=(const itime_mesh_t &mesh) const {
    return !(*this == mesh);
  }

  /// Initialize points in Chebyshev/IR nodes
  void compute_points_cheby() {
    points_.resize(extent());
    points_[0] = 0.0;
    points_[ntau_ - 1] = beta_;
    size_t ncheb = ntau_ - 2;
    for (size_t it = 1; it <= ncheb; ++it) {
      double z = std::cos(M_PI * (it - 0.5) / double(ncheb));
      points_[ntau_ - it - 1] = (z + 1) * beta_ / 2.0;
    }
  }

  // Read points in IR basis
  void compute_points_ir(const std::string &path, const int eta) {
    alps::hdf5::archive inp(path);
    std::vector<double> ir_nodes;
    if (eta == 1) {
      inp["fermi/xsample"] >> ir_nodes;
    } else {
      inp["bose/xsample"] >> ir_nodes;
    }
    inp.close();
    size_t nx = ir_nodes.size();
    ntau_ = nx + 2;
    points_.resize(extent());
    points_[0] = 0.0;
    points_[ntau_ - 1] = beta_;
    for (size_t it = 1; it <= nx; ++it) {
      points_[it] = (ir_nodes[it - 1] + 1) * beta_ / 2.0;
    }
  }

};

///Stream output operator, e.g. for printing to file
inline std::ostream &operator<<(std::ostream &os, const itime_mesh_t &M) {
  os << "# " << "IMAGINARY_TIME_CHEBYSHEV" << " mesh: N: " << M.extent() << " beta: " << M.beta() << std::endl;
  return os;
}

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_IMTIME_MESH_T_H
