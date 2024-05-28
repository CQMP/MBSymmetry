#ifndef SYMMETRYADAPTATION_SPGLIB_ANLYS_H
#define SYMMETRYADAPTATION_SPGLIB_ANLYS_H

#include <memory>

#include "spglib.h"
#include "unit_cell.h"

namespace symmetry_adaptation {

namespace spglib_anlys {

SpglibDataset *get_symmetry_info(const UnitCell &cell, double tol=1e-6);

} // namespace spglib_anlys

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_SPGLIB_ANLYS_H
