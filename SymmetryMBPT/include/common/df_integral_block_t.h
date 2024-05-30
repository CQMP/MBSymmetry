#ifndef SYMMETRYMBPT_DF_INTEGRAL_BLOCK_T_H
#define SYMMETRYMBPT_DF_INTEGRAL_BLOCK_T_H

#include <hdf5.h>
#include <hdf5_hl.h>
#include <alps/hdf5.hpp>

#include "type.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

class df_integral_block_t {
  // prefixes for hdf5
  const std::string rval_ = "VQ";
  const std::string ival_ = "ImVQ";
  const std::string corr_val_ = "EW";
  const std::string corr_bar_val_ = "EW_bar";

public:
  df_integral_block_t(const std::string &path,
                      const symmetry_utils_t &symm_utils,
                      const std::string &prefix="flat_"):
                      base_path_(path), symm_utils_(symm_utils), prefix_(prefix), total_size_(0) {
    hid_t file = H5Fopen((path + "/meta.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (H5LTread_dataset_long(file, "total_size", &total_size_) < 0)
      throw std::logic_error("Fails on reading total_size.");
    H5Fclose(file);
  }

  ~df_integral_block_t() {}

  void read_integrals() {
    vij_Q_.reshape(total_size_);
    read_a_chunk(0, vij_Q_);
  }

  void read_entire(std::complex<double> *Vk1k2_Qij, size_t c_id=0) {
    std::string fname = base_path() + "/" + rval_ + "_" + prefix_ + std::to_string(c_id) + ".h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (H5LTread_dataset_double(file, ("/" + std::to_string(c_id)).c_str(),
                                reinterpret_cast<double *>(Vk1k2_Qij)) < 0) {
      std::cout << "integral file name: " << fname << std::endl;
      throw std::runtime_error("failure reading VQij with chunk id = " + std::to_string(c_id));
    }
    H5Fclose(file);
  }

  void read_a_chunk(size_t c_id, tensor<dcomplex, 1> &V_buffer) {
    std::string fname = base_path() + "/" + rval_ + "_" + prefix_ + std::to_string(c_id) + ".h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (H5LTread_dataset_double(file, ("/" + std::to_string(c_id)).c_str(),
                                reinterpret_cast<double *>(V_buffer.data())) < 0) {
      throw std::runtime_error("failure reading VQij with chunk id = " + std::to_string(c_id));
    }
    H5Fclose(file);
  }

  static void Complex_DoubleToType(const std::complex<double> *in, std::complex<double> *out, size_t size) {
    memcpy(out, in, size * sizeof(std::complex<double>));
  }

  static void Complex_DoubleToType(const std::complex<double> *in, std::complex<float> *out, size_t size) {
    for (int i = 0; i < size; ++i) {
      out[i] = static_cast<std::complex<float> >(in[i]);
    }
  }

  std::pair<int, integral_symmetry_type_e> v_type(size_t k1, size_t k2) {
    size_t idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)
    // determine sign
    int sign = (k1 >= k2)? 1 : -1;
    // determine applied symmetry type
    // by default no symmetries applied
    integral_symmetry_type_e symmetry_type = direct;

    return std::make_pair(sign, symmetry_type);
  }

  // TODO: temporarily store results at all k pairs
  template<typename prec>
  void symmetrize(tensor<prec, 1> &vij_Q_k1k2, const size_t k1, const size_t k2) {
    size_t idx = k1*symm_utils_.nk()+k2;

    long k1k2_wrap = symm_utils_.kpair_slice_offsets(idx);
    size_t k1k2_wrap_size = symm_utils_.kpair_slice_sizes(idx);
    vij_Q_k1k2.vector() = Mcolumn<dcomplex>(vij_Q_.data()+k1k2_wrap, k1k2_wrap_size).cast<prec>();
  }

  // TODO: temporarily store results at all k pairs
  template<typename prec>
  void symmetrize(std::complex<double> *Vk1k2_Qij, tensor<prec, 1> &V, const int k1, const int k2) {
    size_t idx = k1*symm_utils_.nk()+k2;

    long k1k2_wrap = symm_utils_.kpair_slice_offsets(idx);
    size_t k1k2_wrap_size = symm_utils_.kpair_slice_sizes(idx);
    /*
    tensor<dcomplex, 1> V_double_buffer(k1k2_wrap_size);
    memcpy(V_double_buffer.data(), Vk1k2_Qij+k1k2_wrap, k1k2_wrap_size * sizeof(std::complex<double>));
    V.vector() = V_double_buffer.vector().cast<prec>();
     */
    V.reshape(k1k2_wrap_size);
    V.vector() = Mcolumn<dcomplex>(Vk1k2_Qij+k1k2_wrap, k1k2_wrap_size).cast<prec>();
  }

  const std::string &base_path() const {
    return base_path_;
  }

private:

  // Was in base_integral_t in old code
  std::string base_path_;

  // Coulomb integrals stored in density fitting format
  tensor<dcomplex, 1> vij_Q_;

  long total_size_;
  const symmetry_utils_t &symm_utils_;
  const std::string prefix_;
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_DF_INTEGRAL_BLOCK_T_H
