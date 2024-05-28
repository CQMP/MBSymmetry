
#ifndef SYMMETRYADAPTATION_ATOM_H
#define SYMMETRYADAPTATION_ATOM_H

#include <string>

#include "type.h"

namespace symmetry_adaptation {

class Atom {
public:
  Atom(const std::string &element, const ColVector<double, DIM> &pos) : pos_(pos), element_(element) {}

  inline const ColVector<double, DIM> &pos() const { return pos_; }

  inline const std::string &element() const { return element_; }

private:
  const ColVector<double, DIM> pos_;
  const std::string element_;
};

} // namespace symmetry_adaptation

#endif //SYMMETRYADAPTATION_ATOM_H
