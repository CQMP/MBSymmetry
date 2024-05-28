#ifndef SYMMETRYADAPTATION_UTILS_H
#define SYMMETRYADAPTATION_UTILS_H

#include <alps/hdf5.hpp>
#include <alps/type_traits/is_complex.hpp>

#include "type.h"

namespace symmetry_adaptation {

template <typename T>
inline MatrixConstMap<T> Ndarray_MatView(const Ndarray<T> &ndarray, int rows, int cols, int shift = 0) {
  return MatrixConstMap<T>(ndarray.begin() + shift, rows, cols);
}

template <typename T>
inline MatrixMap<T> Ndarray_MatView(Ndarray<T> &ndarray, int rows, int cols, int shift = 0) {
  return MatrixMap<T>(ndarray.begin() + shift, rows, cols);
}

template <typename T>
inline VectorConstMap<T> Ndarray_VecView(const Ndarray<T> &ndarray, int cols, int shift = 0) {
  return VectorConstMap<T>(ndarray.begin() + shift, cols);
}

template <typename T>
inline VectorMap<T> Ndarray_VecView(Ndarray<T> &ndarray, int cols, int shift = 0) {
  return VectorMap<T>(ndarray.begin() + shift, cols);
}

} // namespace symmetry_adaptation


/*
 * Eigen matrix load and dump
 */

// TODO: Save and load column major matrix is not correct with current implementation.

namespace alps {
namespace hdf5 {

namespace adapt = symmetry_adaptation;

template<typename T, int Row, int Col, int M> struct scalar_type<adapt::Matrix<T, Row, Col, M> > {
  typedef typename scalar_type<typename std::remove_cv<T>::type>::type type;
};

template<typename T, int Row, int Col, int M> struct has_complex_elements<adapt::Matrix<T, Row, Col, M> >
    : public has_complex_elements<typename alps::detail::remove_cvr<T>::type> {};


namespace detail {

template<typename T, int Row, int Col, int M> struct get_extent<adapt::Matrix<T, Row, Col, M> > {
  static std::vector<std::size_t> apply(const adapt::Matrix<T, Row, Col, M> & value) {
    using alps::hdf5::get_extent;
    std::vector < std::size_t > result({std::size_t(value.rows()), std::size_t(value.cols())});
    if (value.size()) {
      std::vector < std::size_t > extent(get_extent(*value.data()));
      std::copy(extent.begin(), extent.end(), std::back_inserter(result));
    }
    return result;
  }
};

template<typename T, int Row, int Col, int M> struct set_extent<adapt::Matrix<T, Row, Col, M> > {
  static void apply(adapt::Matrix<T, Row, Col, M> & value, std::vector<std::size_t> const & size) {
    using alps::hdf5::set_extent;
    using alps::hdf5::get_extent;
    if (size.size() > 3 || ((size.size() == 3) && (size[size.size()-1] != 2)))
      throw archive_error("invalid data size" + ALPS_STACKTRACE);
    std::vector < std::size_t > extent(get_extent(*value.data()));
    std::array<size_t, 2> new_size;
    std::copy(size.begin(), size.end() - extent.size(), new_size.begin());
    value.resize(new_size[0], new_size[1]);
  }
};

template<typename T, int Row, int Col, int M> struct get_pointer<adapt::Matrix<T, Row, Col, M> > {
  static typename alps::hdf5::scalar_type<adapt::Matrix<T, Row, Col, M> >::type*
  apply(adapt::Matrix<T, Row, Col, M> &value) {
    using alps::hdf5::get_pointer;
    return get_pointer(*value.data());
  }
};

template<typename T, int Row, int Col, int M> struct get_pointer<const adapt::Matrix<T, Row, Col, M> > {
  static typename alps::hdf5::scalar_type<adapt::Matrix<T, Row, Col, M> >::type const*
  apply(adapt::Matrix<T, Row, Col, M> const &value) {
    using alps::hdf5::get_pointer;
    return get_pointer(*value.data());
  }
};

} // namespace detail

template<typename T, int Row, int Col, int M> void save(
    archive & ar
    , std::string const & path
    , const adapt::Matrix<T, Row, Col, M>& value
    , std::vector<std::size_t> size = std::vector<std::size_t>()
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
  std::vector<std::size_t> extent = get_extent(value);
  std::copy(extent.begin(), extent.end(), std::back_inserter(size));
  std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
  std::fill_n(std::back_inserter(offset), extent.size(), 0);
  ar.write(path, get_pointer(value), size, chunk, offset);
}

template<typename T, int Row, int Col, int M> void load(
    archive & ar
    , std::string const & path
    , adapt::Matrix<T, Row, Col, M> & value
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
  if (ar.is_group(path))
    throw invalid_path("invalid path");
  else {
    if (ar.is_complex(path) != is_complex<T>::value)
      throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
    std::vector<std::size_t> size(ar.extent(path));
    set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));

    std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
    std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
    ar.read(path, get_pointer(value), chunk, offset);
  }
}

} // namespace hdf5
} // namespace alps

/*
 * ndarray load and dump
 */

namespace alps {
namespace hdf5 {

namespace adapt = symmetry_adaptation;

template<typename T> struct scalar_type<adapt::Ndarray<T> > {
  typedef typename scalar_type<typename std::remove_cv<T>::type>::type type;
};

template<typename T> struct has_complex_elements<adapt::Ndarray<T> >
  : public has_complex_elements<typename alps::detail::remove_cvr<T>::type> {};


namespace detail {

template<typename T> struct get_extent<adapt::Ndarray<T> > {
  static std::vector<std::size_t> apply(const adapt::Ndarray<T> & value) {
    using alps::hdf5::get_extent;
    std::vector < std::size_t > result(value.shape());
    if (value.size()) {
      std::vector < std::size_t > extent(get_extent(*value.begin()));
      if (!std::is_scalar<T>::value)
        for (std::size_t i = 1; i < value.size(); ++i)
          if (!std::equal(extent.begin(), extent.end(), get_extent(*(value.begin()+i)).begin())) {
            throw archive_error("non rectangular dataset " + ALPS_STACKTRACE);
          }
      std::copy(extent.begin(), extent.end(), std::back_inserter(result));
    }
    return result;
  }
};

template<typename T> struct set_extent<adapt::Ndarray<T> > {
  static void apply(adapt::Ndarray<T> & value, std::vector<std::size_t> const & size) {
    using alps::hdf5::set_extent;
    using alps::hdf5::get_extent;
    std::vector < std::size_t > extent(get_extent(*value.begin()));
    std::vector<size_t> new_size(0);
    new_size.insert(new_size.begin(), size.begin(), size.end() - extent.size());
    value.reshape(new_size);
  }
};

template<typename T> struct get_pointer<adapt::Ndarray<T> > {
  static typename alps::hdf5::scalar_type<adapt::Ndarray<T> >::type * apply(adapt::Ndarray<T> & value) {
    using alps::hdf5::get_pointer;
    return get_pointer(*value.begin());
  }
};

template<typename T> struct get_pointer<const adapt::Ndarray<T> > {
  static typename alps::hdf5::scalar_type<adapt::Ndarray<T> >::type const * apply(adapt::Ndarray<T> const & value) {
    using alps::hdf5::get_pointer;
    return get_pointer(*value.begin());
  }
};

} // namespace detail

template<typename T> void save(
  archive & ar
  , std::string const & path
  , const adapt::Ndarray<T>& value
  , std::vector<std::size_t> size = std::vector<std::size_t>()
  , std::vector<std::size_t> chunk = std::vector<std::size_t>()
  , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
  std::vector<std::size_t> extent = get_extent(value);
  std::copy(extent.begin(), extent.end(), std::back_inserter(size));
  std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
  std::fill_n(std::back_inserter(offset), extent.size(), 0);
  ar.write(path, get_pointer(value), size, chunk, offset);
}

template<typename T> void load(
  archive & ar
  , std::string const & path
  , adapt::Ndarray<T> & value
  , std::vector<std::size_t> chunk = std::vector<std::size_t>()
  , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
  if (ar.is_group(path))
    throw invalid_path("invalid path");
  else {
    if (ar.is_complex(path) != is_complex<T>::value)
      throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
    std::vector<std::size_t> size(ar.extent(path));
    set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));

    std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
    std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
    ar.read(path, get_pointer(value), chunk, offset);
  }
}

} // namespace hdf5
} // namespace alps

#endif //SYMMETRYADAPTATION_UTILS_H
