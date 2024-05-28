#ifndef SYMMETRYADAPTATION_PARAMS_H
#define SYMMETRYADAPTATION_PARAMS_H

#include <iostream>
#include <fstream>

#include <alps/params.hpp>

namespace symmetry_adaptation {

inline void define_solid_params(alps::params &par) {

  par.define<std::string>("kmesh_file", "NONE", "Name of h5 file for k mesh");
  par.define<std::string>("tvec_file", "NONE", "Name of translation vector file");
  par.define<std::string>("kspace_convention", "pyscf", "Convention of k mesh");
  par.define<int>("q", 1, "Number of points in one direction of unit cell");
  par.define<bool>("read_k_mesh", false, "Read k mesh file");
  par.define<int>("phase_type", 1, "Phase factor type in k space");
  par.define<std::string>("adapt_type", "space", "Solid symmetry adaptation type, can be space or symmorphic");
}

inline void define_params(alps::params &par) {

  // Inputs
  par.define<std::string>("xyz_file", "Name of geometry file");
  par.define<std::string>("basis_file", "Name of basis file");
  par.define<int>("group_nr", 0, "Number of the point group (1-31)");
  par.define<int>("max_n", 12, "Maximum n index when defining orbitals");
  par.define<double>("tol", 1e-12, "Numerical tolerance");
  par.define<bool>("verbose", true, "Print information");
  par.define<std::string>("convention", "pyscf", "Convention of real spherical harmonics");
  par.define<bool>("use_lapack", false, "Use explicit lapack call when solving eigenvalue problem");

  par.define<bool>("diagonalization", true, "Use simultaneous diagonalization to compute transform matrix");
  par.define<bool>("time_reversal", false, "Consider time reversal relation when computing transform matrix");

  par.define<std::string>("calc_type", "solid", "Symmetry adaptation for molecule or solid");

  if (par["calc_type"] == "solid") {
    define_solid_params(par);
  }

  // Output
  par.define<std::string>("output_file", "Name of output h5 file for transformation matrix");

  // If requested, we print the help message, which is constructed from the
  // information we gave when defining the parameters.
  if (par.help_requested(std::cout)) {
    exit(0);
  }

  if (par.has_missing(std::cout)) {
    exit(1);
  }
}

// Temporarily put it here, will move to other places later
inline std::string read_xyz_file(const std::string &path, bool verbose=true) {

  if (verbose)
    std::cout << "reading text file information: " << path << std::endl;
  std::ifstream xyzfile(path.c_str());
  if (!xyzfile.good()) throw std::runtime_error("file could not be opened.");

  std::stringstream buffer;
  buffer << xyzfile.rdbuf();
  std::cout << std::endl;
  return buffer.str();
}

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_PARAMS_H
