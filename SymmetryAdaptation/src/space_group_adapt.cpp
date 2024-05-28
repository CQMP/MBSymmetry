#include "basis.h"
#include "dirac_character.h"
#include "k_mesh.h"
#include "k_space_structure.h"
#include "unit_cell.h"
#include "params.h"
#include "translation_vector.h"
#include "type.h"
#include "diagonalization.h"
#include "space_group_rep.h"

int main(int argc, char **argv) {
  using namespace symmetry_adaptation;

  alps::params par(argc, argv);
  par["calc_type"] = "solid";
  define_params(par);

  bool verbose = par["verbose"];
  double tol = par["tol"];
  bool use_lapack = par["use_lapack"];
  int max_n = par["max_n"];

  bool diag = par["diagonalization"];
  bool time_reversal = par["time_reversal"];

  // Set up molecule
  Basis basis(max_n);
  basis.read(par["basis_file"].as<std::string>(), verbose);

  std::string tvec_file = read_xyz_file(par["tvec_file"].as<std::string>());
  TranslationVector tvec(tvec_file, verbose);
  std::string xyz_file = read_xyz_file(par["xyz_file"].as<std::string>(), verbose);
  UnitCell unit_cell(tvec, xyz_file, basis, verbose);

  alps::hdf5::archive output_file(par["output_file"].as<std::string>(), "w");

  int q = par["q"];
  KMesh k_mesh(q, tvec);
  bool read_k_mesh = par["read_k_mesh"];
  if (read_k_mesh) {
    k_mesh.read_k_mesh(par["kmesh_file"].as<std::string>());
  }
  else {
    k_mesh.generate_k_mesh(par["kspace_convention"].as<std::string>());
  }
  k_mesh.save(output_file);

  int point_group_number = par["group_nr"];  // default is zero

  WignerD wigner_d(point_group_number, par["convention"].as<std::string>());
  Ndarray<dcomplex> orbital_rep;
  std::unique_ptr<KSpaceStructure> k_struct = nullptr;

  SpaceGroup group(tol);
  group.get_space_group_info(unit_cell, diag);
  if (verbose) {
    std::cout << "space group number is: " << group.space_group_number() << std::endl;
    std::cout << "space group is a symmorphic group: " << std::boolalpha << group.symmorphic() << std::endl;
    std::cout << "space group has inversion operation: " << std::boolalpha << group.has_inversion() << std::endl;
  }
  wigner_d.compute(group);
  k_struct = std::make_unique<KSpaceStructure>(k_mesh, group, tvec, tol, verbose, diag, time_reversal);
  orbital_rep = space_group_representation::generate_representation((*k_struct), group,
                                                                    unit_cell, wigner_d, tol, verbose);
  group.save(output_file);
  wigner_d.save(output_file);
  k_struct->save(output_file);
  output_file["/KspaceORep"] << orbital_rep;

  if (diag) {
    diagonalization::diagonalize_proj_dirac_characters(k_struct, group, orbital_rep, output_file,
                                                       use_lapack, tol, verbose, time_reversal);
  }
}
