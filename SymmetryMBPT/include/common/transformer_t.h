#ifndef SYMMETRYMBPT_TRANSFORMER_T_H
#define SYMMETRYMBPT_TRANSFORMER_T_H

#include "imtime_mesh_t.h"
#include "type.h"
#include "params_t.h"

namespace symmetry_mbpt {

/**
 * @brief Class for imaginary time transforms.
 * Perform Fourier transform between imaginary time and Matsubara frequency domains using intermediate representation.
 */
class transformer_t {
public:
  explicit transformer_t(const params_t &p): transformer_t(p.ni, p.beta, p.tnc_f_file, p.tnc_b_file, p.IR) {}
  transformer_t(int ni, int beta, const std::string &tnc_f_file,
                const std::string &tnc_b_file, bool IR=true):
                nts_(ni+2), ncheb_(ni), IR_(IR),
                tau_mesh_(ni + 2, beta, 1, IR, tnc_f_file),
                tau_mesh_B_(ni + 2, beta, 0, IR, tnc_b_file),
                beta_(beta) {
    if (!IR_) {
      if (ncheb_ % 2 != 0) throw std::logic_error("number of Fermionic Chebyshev should be even");
      ncheb_b_  = ni - 1;
      nts_b_ = ncheb_b_  + 2;
      read_wsample(tnc_f_file, tnc_b_file, wsample_fermi_, wsample_bose_);
      // Read Tnc_ and Tnc_B_
      read_chebyshev(tnc_f_file, tnc_b_file);
      // Compute Ttc_, Ttc_B_. Tct_, and Tct_B_
      init_chebyshev_grid();
    } else {
      read_wsample(tnc_f_file, tnc_f_file, wsample_fermi_, wsample_bose_);
      nw_ = wsample_fermi_.size();
      nw_b_ = wsample_bose_.size();
      nts_b_ = tau_mesh_B_.extent();
      ncheb_b_  = nts_b_ - 2;
      // Read Tnc_, Tnc_B_, Ttc_, Ttc_B_, Tct_, and Tct_B_
      read_trans_ir(tnc_f_file);
    }
  }

  /**
   * @param n   - [INPUT] Matsubara frequency number, omega(n) = iw_n
   * @param eta - [INPUT] eta-parameter:
   *      0 - bosonic Matsubara
   *      1 - fermionic Matsubara
   *
   * @return value of w_n-th Matsubara frequency point
   */
  inline std::complex<double> omega(long n, int eta) const {
    return std::complex<double>(0.0, (2.0 * (n) + eta) * M_PI / beta_);
  }

  inline const std::vector<long> &wsample_bose() const { return wsample_bose_; }

  inline const std::vector<long> &wsample_fermi() const { return wsample_fermi_; }

  /**
   * Transform the tensor represented in fermionic imaginary time into bosonic Matsubara frequency through the intermediate representation
   *
   * @tparam N  - tensor dimension
   * @param F_t - [INPUT]  tensor in fermionic imaginary time
   * @param F_w - [OUTPUT] tensor in bosonic Matsubara frequency
   */
  template<size_t N, typename S1, typename S2>
  void tau_f_to_w_b(const tensor_base<dcomplex, N, S1> &F_t, tensor_base<dcomplex, N, S2> &F_w) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_w = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
    assert(dim_t == dim_w);

    MMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim_w);
    CMMatrixXcd f_t(F_t.data() + dim_t, ncheb_, dim_t);
    f_w = Tnt_BF_ * f_t;
  }

  /**
   * Transform the tensor represented in bosonic Matsubara frequency into fermionic imaginary time through the intermediate representation
   *
   * @tparam N  - tensor dimension
   * @param F_w - [INPUT] tensor in bosonic Matsubara frequency
   * @param F_t - [OUTPUT]  tensor in fermionic imaginary time
   */
  template<size_t N, typename S1, typename S2>
  void w_b_to_tau_f(const tensor_base<dcomplex, N, S1> &F_w, tensor_base<dcomplex, N, S2> &F_t) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_w = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
    assert(dim_t == dim_w);

    MMatrixXcd f_t(F_t.data(), F_t.shape()[0], dim_t);
    CMMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim_w);
    f_t = Ttn_FB_ * f_w;
  }

  /**
   * Intermediate step of non-uniform Fourier transformation. Convert object in Chebyshev/IR representation into Matsubara frequency representation
   * @param F_c - [INPUT] Object in Chebyshev representation
   * @param F_w - [OUTPUT] Object in Matsubara frequency representation
   * @param eta - [INPUT] statistics
   */
  template<size_t N>
  void chebyshev_to_matsubara(const tensor<dcomplex, N> &F_c, tensor<dcomplex, N> &F_w, int eta) const {
    size_t dim1 = std::accumulate(F_c.shape().begin() + 1, F_c.shape().end(), 1ul, std::multiplies<size_t>());
    // f_w(ncheb, dim1)
    MMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim1);
    // f_c(ncheb, dim1)
    CMMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim1);
    // Tnl(nwn=ncheb, ncheb)
    f_w = (eta ? Tnc_ : Tnc_B_) * f_c;
  }

  /**
   * Intermediate step of non-uniform Fourier transformation. Convert object in Matsubara frequency representation into Chebyshev/IR representation
   * @param F_w - [INPUT] Object in Matsubara frequency representation
   * @param F_c - [OUTPUT] Object in Chebyshev representation
   * @param eta - [INPUT] statistics
   */
  template<size_t N>
  void matsubara_to_chebyshev(const tensor<dcomplex, N> &F_w, tensor<dcomplex, N> &F_c, int eta) const {
    size_t dim1 = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
    // f_c(ncheb, dim1)
    MMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim1);
    // f_w(nw, dim1)
    CMMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim1);
    // Tnl(nw, ncheb)
    f_c = (eta ? Tcn_ : Tcn_B_) * f_w;
  }

  /**
   * Intermediate step of Chebyshev convolution. Convert object in Chebyshev/IR representation into tau axis
   */
  template<size_t N>
  void
  chebyshev_to_tau(const tensor<dcomplex, N> &F_c, tensor<dcomplex, N> &F_t, int eta, bool dm_only = false) const {
    size_t dim_c = std::accumulate(F_c.shape().begin() + 1, F_c.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_t = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    // f_c(ncheb, dim_c)
    CMMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim_c);
    // f_t(nts, dim_t)
    MMatrixXcd f_t(F_t.data(), F_t.shape()[0], dim_t);
    if (eta == 1) {
      if (!dm_only) {
        f_t = Ttc_ * f_c;
      } else {
        f_t = Ttc_.row(nts_ - 1) * f_c;
      }
    } else {
      if (!dm_only) {
        f_t = Ttc_B_ * f_c;
      } else {
        f_t = Ttc_B_.row(nts_b_ - 1) * f_c;
      }
    }
  }

  /**
   * compute transform from imaginary time to intermediate basis for the input object.
   *
   * We use the following matrix multiplication scheme to go from imaginary time to Chebyshev polynomials
   * F_c(c, k, i, j) = _Fct(c, t) * F_t(t, k, i, j)
   *
   * Where _Fct is the transition matrix
   *
   * @param F_t - [INPUT] Object in imaginary time
   * @param F_c - [OUTPUT] Object in Chebyshev basis
   * @param eta - [INPUT] statistics
   */
  template<size_t N>
  void tau_to_chebyshev(const tensor<dcomplex, N> &F_t, tensor<dcomplex, N> &F_c, int eta = 1) const {
    tau_to_chebyshev(tensor_view<dcomplex, N>(F_t), F_c, eta);
  }

  template<size_t N>
  void tau_to_chebyshev(const tensor_view<dcomplex, N> &F_t, tensor<dcomplex, N> &F_c, int eta = 1) const {
    // Calculate coefficients in Chebyshev nodes
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    //size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
    // Chebyshev tensor as rectangular matrix
    MMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim1);
    if (!IR_) {
      // Imaginary-time tensor as rectangular matrix
      // f_t(t, dim1)
      CMMatrixXcd f_t(F_t.data(), F_t.shape()[0], dim1);
      // _Tct(ncheb_, t)
      f_c = (eta == 1) ? Tct_ * f_t : Tct_B_ * f_t;
    } else {
      // If using ir basis, Tct_ = (nl, nx) instead of (nl, nx+2)
      if (eta == 1) {
        CMMatrixXcd f_t(F_t.data() + dim1, ncheb_, dim1);
        f_c = Tct_ * f_t;
      } else {
        CMMatrixXcd f_t(F_t.data() + dim1, ncheb_b_, dim1);
        f_c = Tct_B_ * f_t;
      }
    }
  }

  template<size_t N, typename storage>
  void tau_to_chebyshev_c(const tensor_base<dcomplex, N, storage> &F_t,
                          tensor<dcomplex, N> &F_c, size_t ic, int eta = 1) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    assert(ic < (eta == 1 ? ncheb_ : ncheb_b_));
    // Calculate coefficients in Chebyshev nodes
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    //size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
    // Chebyshev tensor as rectangular matrix
    MMatrixXcd f_c(F_c.data(), F_c.shape()[0], dim1);
    int n = (eta == 1 ? Tct_ : Tct_B_).rows();
    int m = (eta == 1 ? Tct_ : Tct_B_).cols();
    MatrixXcd Tct = (eta == 1 ? Tct_ : Tct_B_).block(ic, 0, 1, n);
    // If using ir basis, Tct_ = (nl, nx) instead of (nl, nx+2)
    CMMatrixXcd f_t(F_t.data() + dim1, eta == 1 ? ncheb_ : ncheb_b_, dim1);
    f_c = Tct * f_t;
  }

  template<size_t N, typename storage>
  void tau_to_omega(const alps::numerics::detail::tensor_base<dcomplex, N, storage> &F_t, tensor<dcomplex, N> &F_w,
                    int eta = 1) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    // Calculate coefficients in Chebyshev nodes
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    //size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
    // Chebyshev tensor as rectangular matrix
    MMatrixXcd f_w(F_w.data(), F_w.shape()[0], dim1);
    if (eta == 1) {
      CMMatrixXcd f_t(F_t.data() + dim1, ncheb_, dim1);
      f_w = Tnt_ * f_t;
    } else {
      CMMatrixXcd f_t(F_t.data() + dim1, ncheb_b_, dim1);
      f_w = Tnt_B_ * f_t;
    }
  }

  /*template<size_t N, typename storage>
  typename std::enable_if<(N > 3), void>::type tau_to_omega_wsk(const tensor_base<dcomplex, N, storage> &F_t,
                                                                tensor<dcomplex, N-3> &F_w, size_t w, size_t s,
                                                                size_t k, int eta = 1) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    // Calculate coefficients in Chebyshev nodes
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim2 = std::accumulate(F_t.shape().begin() + 2, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim3 = std::accumulate(F_t.shape().begin() + 3, F_t.shape().end(), 1ul, std::multiplies<size_t>());

    size_t n = (eta == 1 ? Tnt_ : Tnt_B_).rows();
    size_t m = (eta == 1 ? Tnt_ : Tnt_B_).cols();
    assert(w < n);
    MatrixXcd Tnt = (eta == 1 ? Tnt_ : Tnt_B_).block(w, 0, 1, m);

    MMatrixXcd f_w(F_w.data(), 1, dim3);
    Eigen::Map<const MatrixXcd, 0, Eigen::OuterStride<> > f_t(F_t.data() + dim1 + s * dim2 + k * dim3, m, dim3,
                                                              Eigen::OuterStride<>(dim1));
    f_w = Tnt * f_t;
  }*/

  template<size_t N, typename storage>
  typename std::enable_if<(N > 1), void>::type tau_to_omega_single(const tensor_base<dcomplex, N, storage> &F_t,
                                                                   tensor<dcomplex, N-1> &F_w,
                                                                   size_t w, int eta = 1) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    size_t n = (eta == 1 ? Tnt_ : Tnt_B_).rows();
    size_t m = (eta == 1 ? Tnt_ : Tnt_B_).cols();
    assert(w < n);
    MatrixXcd Tnt = (eta == 1 ? Tnt_ : Tnt_B_).block(w, 0, 1, m);

    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(F_t.shape().begin() + 1, F_t.shape().end(), 1ul, std::multiplies<size_t>());
    MMatrixXcd f_w(F_w.data(), 1, dim1);
    CMMatrixXcd f_t(F_t.data() + dim1, m, dim1);
    f_w = Tnt * f_t;
  };

  template<size_t N>
  void omega_to_tau(const tensor<dcomplex, N> &F_w, tensor<dcomplex, N> &F_t, int eta = 1) const {
    if (!IR_) {
      throw std::logic_error("Chebyshev is not implemented");
    }
    // Calculate coefficients in Chebyshev nodes
    // Dimension of the rest of arrays
    size_t dim1 = std::accumulate(F_w.shape().begin() + 1, F_w.shape().end(), 1ul, std::multiplies<size_t>());
    //size_t dim1 = F_t.shape()[1]*F_t.shape()[2]*F_t.shape()[3]*F_t.shape()[4];
    // Chebyshev tensor as rectangular matrix
    MMatrixXcd f_t(F_t.data(), F_t.shape()[0], dim1);
    if (eta == 1) {
      CMMatrixXcd f_w(F_w.data(), nw_, dim1);
      f_t = Ttn_ * f_w;
    } else {
      CMMatrixXcd f_w(F_w.data(), nw_b_, dim1);
      f_t = Ttn_B_ * f_w;
    }
  }

  const MatrixXcd &Ttn() const { return Ttn_; }
  const MatrixXcd &Tnt() const { return Tnt_; }

  const MatrixXcd &Tnt_BF() const { return Tnt_BF_; }
  const MatrixXcd &Ttn_FB() const { return Ttn_FB_; }

  int nw() const { return nw_; }
  int nw_b() const { return nw_b_; }

private:

  // Frequency samplings
  std::vector<long> wsample_fermi_;
  std::vector<long> wsample_bose_;

  // Fourier normalization factor
  double nkpw_;

  // number of imaginary time points
  size_t nts_;
  size_t nts_b_;

  // number of Chebyshev polynomials
  size_t ncheb_;
  size_t ncheb_b_;

  // number of frequency points
  size_t nw_;
  size_t nw_b_;

  // IR basis
  bool IR_;

  // tau meshes
  itime_mesh_t tau_mesh_;
  itime_mesh_t tau_mesh_B_;

  // inverse temperature
  double beta_;

  // Transformation matrices
  // Fermi
  // from ir basis to Matsubara
  MatrixXcd Tnc_;
  // from Matsubara to ir
  MatrixXcd Tcn_;
  // from ir to tau
  MatrixXd Ttc_;
  // from fermionic ir to bosonic tau grid
  MatrixXd Ttc_other_;
  // from tau to ir
  MatrixXd Tct_;
  // Bose
  MatrixXcd Tnc_B_;
  MatrixXcd Tcn_B_;
  MatrixXd Ttc_B_;
  // from bosonic ir to fermionic tau grid
  MatrixXd Ttc_B_other_;
  MatrixXd Tct_B_;

  MatrixXcd Tnt_;
  MatrixXcd Ttn_;
  MatrixXcd Tnt_B_;
  MatrixXcd Ttn_B_;

  // Transform from Bosonic frequency to Fermionic imaginary time
  MatrixXcd Ttn_FB_;
  // Transform from Fermionic imaginary time to Bosonic frequency
  MatrixXcd Tnt_BF_;

  /**
   * Initialize transformation matrix between tau to Chebyshev representation
   */
  void init_chebyshev_grid();

  /**
   * Read Chebyshev transformation matrix from Chebyshev representation to Matsubara axis
   * @param path   - [INPUT] path to Fermionic precomputed transformation matrix
   * @param path_B - [INPUT] path to Bosonic precomputed transformation matrix
   */
  void read_chebyshev(const std::string &path, const std::string &path_B);

  /**
   * Read IR transformation matrix from IR representation to Matsubara axis
   * @param path   - [INPUT] path to Fermionic and Bosonic precomputed transformation matrices
   */
  void read_trans_ir(const std::string &path);

  void read_trans_ir_statistics(alps::hdf5::archive &tnl_file, int eta, MatrixXcd &Tnc_out, MatrixXcd &Tcn_out, MatrixXd &Ttc_out,
                                MatrixXd &Tct_out);


  /**
   * Read frequency sampling points
   * @param path - [INPUT] path to sampling points
   * @param path_b    - [INPUT] path to Bosonic sampling points
   * @param wsample   - [OUTPUT] Fermionic sampling points
   * @param wsample_b - [OUTPUT] Bosonic sampling points
   */
  void read_wsample(const std::string &path, const std::string &path_b, std::vector<long> &wsample, std::vector<long> &wsample_b);

  /**
   * Trasnformation in Chebyshev representations between even and odd points.
   * Fermionoc to Bosonic: ncheb(even) to ncheb-1(odd)
   * Bosonic to Fermionic: ncheb-1(odd) to ncheb(even)
   * @param eta - [INPUT] starting statistics
   */
  void fermi_boson_trans_cheby(const tensor<dcomplex, 4> &F_t_before, tensor<dcomplex, 4> &F_t_after, int eta);

  /**
   * Transformation in IR representation between Fermionic and Bosonic grids.
   * @param eta - [INPUT] starting statistics
   */
  void fermi_boson_trans_ir(const tensor<dcomplex, 4> &F_t_before, tensor<dcomplex, 4> &F_t_after, int eta);
};

} // namespace symmetry_mbpt


#endif //SYMMETRYMBPT_TRANSFORMER_T_H
