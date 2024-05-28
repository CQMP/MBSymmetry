#ifndef SYMMETRYADAPTATION_DIAGONALIZATION_H
#define SYMMETRYADAPTATION_DIAGONALIZATION_H

#include "k_space_structure.h"
#include "dirac_character.h"
#include "simul_diag.h"

namespace symmetry_adaptation {

namespace diagonalization {

void diagonalize_dirac_characters(const std::unique_ptr<KSpaceStructure> &k_struct,
                                  Ndarray<dcomplex> &orbital_rep,
                                  alps::hdf5::archive &output_file,
                                  bool use_lapack=false,
                                  double tol=1e-6, bool verbose=true,
                                  bool time_reversal=false);

void diagonalize_proj_dirac_characters(const std::unique_ptr<KSpaceStructure> &k_struct,
                                       const SpaceGroup &group,
                                       Ndarray<dcomplex> &orbital_rep,
                                       alps::hdf5::archive &output_file,
                                       bool use_lapack=false,
                                       double tol=1e-6, bool verbose=true,
                                       bool time_reversal=false);

// This function will update the U matrix in dirac character
void apply_time_reversal_relation(DiracCharacter &dirac, double tol);

} // namespace diagonalization

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_DIAGONALIZATION_H
