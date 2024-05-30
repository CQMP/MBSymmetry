#ifndef SYMMETRYMBPT_DF_INTEGRAL_T_H
#define SYMMETRYMBPT_DF_INTEGRAL_T_H

#include <hdf5.h>
#include <hdf5_hl.h>

#include "type.h"
#include "symmetry_utils.h"

namespace symmetry_mbpt {

/**
  * @brief Integral class read Density fitted 3-center integrals from a HDF5 file, given by the path argument
  */
class df_integral_t {
  // prefixes for hdf5
  const std::string rval_ = "VQ";
  const std::string ival_ = "ImVQ";
  const std::string corr_val_ = "EW";
  const std::string corr_bar_val_ = "EW_bar";

public:
  df_integral_t(const std::string &path, int nao, int NQ,
                const symmetry_utils_t &symm_utils,
                bool symm=true, std::string prefix="",
                integral_reading_type reading_type = chunks) :
    base_path_(path), vij_Q_(1, NQ, nao, nao), k0_(-1),
    current_chunk_(-1), chunk_size_(0),
    symm_utils_(symm_utils), reading_type_(reading_type),
    symm_(symm), prefix_(prefix) {
    hid_t file = H5Fopen((path + "/meta.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (H5LTread_dataset_long(file, "chunk_size", &chunk_size_) < 0)
      throw std::logic_error("Fails on reading chunk_size.");
    H5Fclose(file);
    vij_Q_.reshape(chunk_size_, NQ, nao, nao);
  }

  ~df_integral_t() {};

  /**
   * Read next part of the interaction integral from
   * @param k1
   * @param k2
   * @param type
   */
  void read_integrals(size_t k1, size_t k2, IntegralType type = first_integral) {

    // Find corresponding index for k-pair (k1,k2). Only k-pair with k1 > k2 will be stored.
    size_t idx;
    if (symm_) {
      idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)
    }
    else {
      idx = k1 * symm_utils_.nk() + k2;
    }
    long idx_red = symm_utils_.irre_pos_kpair(idx);
    if ((idx_red / chunk_size_) == current_chunk_) return; // we have data cached

    current_chunk_ = idx_red / chunk_size_;

    size_t c_id = current_chunk_ * chunk_size_;
    read_a_chunk(c_id, vij_Q_);
  }

  /**
   * Read the entire 3-indx Coulomb intergrals without MPI-parallel.
   * @param prec - flaot/double
   * @param Vk1k2_Qij - A pointer to 3-index Coulomb integrals
   */
  template<typename prec>
  void read_entire(std::complex<prec> *Vk1k2_Qij) {
    const int NQ = vij_Q_.shape()[1];
    const int nao = vij_Q_.shape()[2];
    size_t num_kpair_stored = symm_utils_.num_kpair_stored();
    size_t last_chunk_id = (num_kpair_stored / chunk_size_) * chunk_size_;
    tensor<dcomplex, 4> V_Qij_chunk(chunk_size_, NQ, nao, nao);
    for (std::size_t c_id = 0; c_id < num_kpair_stored; c_id += chunk_size_) {
      size_t shift = c_id * NQ * nao * nao;
      size_t element_counts = (c_id != last_chunk_id) ? chunk_size_ * NQ * nao * nao
                                                      : (num_kpair_stored - c_id) * NQ * nao * nao;
      read_a_chunk(c_id, V_Qij_chunk);

      Complex_DoubleToType(V_Qij_chunk.data(), Vk1k2_Qij + shift, element_counts);
    }
  }

  void read_a_chunk(size_t c_id, tensor<dcomplex, 4> &V_buffer) {
    std::string fname = base_path() + "/" + rval_ + "_" + prefix_ + std::to_string(c_id) + ".h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (H5LTread_dataset_double(file, ("/" + std::to_string(c_id)).c_str(),
                                reinterpret_cast<double *>(V_buffer.data())) < 0) {
      throw std::runtime_error("failure reading VQij with chunk id = " + std::to_string(c_id));
    }
    H5Fclose(file);
    /*alps::hdf5::archive file(fname, "r");
    file["/" + std::to_string(c_id)] >> V_buffer;
    file.close();*/
  }

  static void Complex_DoubleToType(const std::complex<double> *in, std::complex<double> *out, size_t size) {
    memcpy(out, in, size * sizeof(std::complex<double>));
  }

  static void Complex_DoubleToType(const std::complex<double> *in, std::complex<float> *out, size_t size) {
    for (int i = 0; i < size; ++i) {
      out[i] = static_cast<std::complex<float> >(in[i]);
    }
  }

  /**
   * Determine the type of symmetries for the integral based on the current k-points
   *
   * @param k1 incomming k-point
   * @param k2 outgoing k-point
   * @return A pair of sign and type of applied symmetry
   */
  std::pair<int, integral_symmetry_type_e> v_type(size_t k1, size_t k2) {
    size_t idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)
    // determine sign
    int sign = (k1 >= k2)? 1 : -1;
    // determine applied symmetry type
    // by default no symmetries applied
    integral_symmetry_type_e symmetry_type = direct;

    return std::make_pair(sign, symmetry_type);
  }

  /**
   * Extract V(Q, i, j) with given (k1, k2) from chunks of integrals (_vij_Q)
   * @tparam prec
   * @param vij_Q_k1k2
   * @param k1
   * @param k2
   */
  template<typename prec>
  void symmetrize(tensor<prec, 3> &vij_Q_k1k2, const size_t k1, const size_t k2) {
    int NQ = vij_Q_.shape()[1];

    int k1k2_wrap;
    std::pair<int, integral_symmetry_type_e> vtype;
    if (symm_) {
      k1k2_wrap = wrap(k1, k2);
      vtype = v_type(k1, k2);
    }
    else {
      k1k2_wrap = k1 * symm_utils_.nk() + k2;
      vtype = std::make_pair(1, direct);
    }

    if (vtype.first < 0) {
      for (int Q = 0; Q < NQ; ++Q) {
        vij_Q_k1k2(Q).matrix() = vij_Q_(k1k2_wrap, Q).matrix().transpose().conjugate().cast<prec>();
      }
    } else {
      for (int Q = 0; Q < NQ; ++Q) {
        vij_Q_k1k2(Q).matrix() = vij_Q_(k1k2_wrap, Q).matrix().cast<prec>();
      }
    }
  }

  int wrap(int k1, int k2, integral_reading_type read_type = chunks) {
    size_t idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)

    int idx_red = symm_utils_.irre_pos_kpair(idx);
    return (read_type==chunks)? idx_red % chunk_size_ : idx_red;
  }

  void reset() {
    current_chunk_ = -1;
    k0_ = -1;
  }

  const std::string &base_path() const {
    return base_path_;
  }

private:

  // Was in base_integral_t in old code
  std::string base_path_;

  // Coulomb integrals stored in density fitting format
  tensor<dcomplex, 4> vij_Q_;
  // G=0 correction to coulomb integral stored in density fitting format
  // TODO: check these two tensors for ewald correction
  tensor<dcomplex, 3> v0ij_Q_;
  tensor<dcomplex, 3> v_bar_ij_Q_;

  bool exch_;
  // current leading index
  int k0_;
  long current_chunk_;
  long chunk_size_;
  const symmetry_utils_t &symm_utils_;

  integral_reading_type reading_type_;
  std::string prefix_;
  bool symm_;
};

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_DF_INTEGRAL_T_H
