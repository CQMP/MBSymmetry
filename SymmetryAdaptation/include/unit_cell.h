#ifndef SYMMETRYADAPTATION_UNIT_CELL_H
#define SYMMETRYADAPTATION_UNIT_CELL_H

#include <utility>

#include "molecule.h"
#include "translation_vector.h"

namespace symmetry_adaptation {

class UnitCell : public Molecule {

public:

  UnitCell(const TranslationVector& vec) : trans_vec_(vec) {};

  UnitCell(const TranslationVector& vec,
           const std::string &xyz_str, bool verbose = true) : Molecule(xyz_str, verbose), trans_vec_(vec) {};

  UnitCell(const TranslationVector& vec, const std::string &xyz_str,
           const Basis &basis, bool verbose = true) : Molecule(xyz_str, basis, verbose), trans_vec_(vec) {};

  ~UnitCell() override = default;

  inline const TranslationVector &trans_vec() const { return trans_vec_; }

private:

  const TranslationVector &trans_vec_;
};

}

#endif //SYMMETRYADAPTATION_UNIT_CELL_H
