#ifndef SYMMETRYMBPT_SC_LOOP_BLOCK_T_H
#define SYMMETRYMBPT_SC_LOOP_BLOCK_T_H

#include "type.h"
#include "timing.h"
#include "params_t.h"
#include "symmetry_utils.h"
#include "transformer_t.h"
#include "solver_block_t.h"
#include "solver_factory_t.h"
#include "sc_type.h"

namespace symmetry_mbpt {
  /**
  * @brief sc_loop class perform the main self-consistency loop for self-energy evaluation
  */
  class sc_loop_block_t {
  public:
    sc_loop_block_t(MPI_Comm comm, params_t &p) : params_(p), symm_utils_(p), npoly_(p.ni), nts_(p.ni + 2), IR_(p.IR),
                                                  nk_(p.nk), ink_(symm_utils_.ink()), ns_(p.ns),
                                                  F_k_(ns_, symm_utils_.flat_ao_size()),
                                                  S_k_(ns_, symm_utils_.flat_ao_size()),
                                                  H_k_(ns_, symm_utils_.flat_ao_size()),
                                                  Bz_(p.Bz),
                                                  beta_(p.beta), mu_(p.mu), const_density_(p.CONST_DENSITY),
                                                  nel_(p.nel_cell), tol_(p.tol),
                                                  input_path_(p.input_file),
                                                  tau_mesh_(nts_, p.beta, 1, p.IR, p.tnc_f_file), ft_(p),
                                                  G_tau_(nullptr, nts_, ns_, symm_utils_.flat_ao_size()),
                                                  Selfenergy_(nullptr, nts_, ns_, symm_utils_.flat_ao_size()),
                                                  dmr_(ns_, symm_utils_.flat_ao_size()), world_comm_(comm),
                                                  solver_(solver_factory_t::construct_solver(comm, p, symm_utils_, ft_, G_tau_, Selfenergy_)),
                                                  scf_type_(p.scf_type), sc_type_(nullptr) {
    MPI_Comm_rank(comm, &world_rank_);
    MPI_Comm_size(comm, &world_size_);
    initialize_shmem_communicators();
    allocate_shared_memory();

    p.ink = ink_;  // just in case we give wrong ink in parameter
    if (p.sc_type == Dyson) {
      sc_type_ = new dyson_sc_block_t(p, ft_, symm_utils_);
    }
    else
      throw std::logic_error("only Dyson sc_type has been implemented");

    if (!world_rank_) {
      std::cerr<<"####################################################"<<std::endl;
      double gf_mem = nts_ * ns_ * symm_utils_.flat_ao_size() * sizeof(std::complex<double>) / (1024. * 1024. * 1024.);
      double shared_Coul_mem = symm_utils_.flat_V_size() * sizeof(std::complex<double>) / (1024. * 1024. * 1024.);
      double static_1e_mem = (F_k_.size() * 6) * sizeof(std::complex<double>) / (1024. * 1024. * 1024);
      double shared_P_mem = nts_ * symm_utils_.flat_aux_size() * sizeof(std::complex<double>) / (1024. * 1024. * 1024.);

      int total_size = 0;
      /*hid_t file = H5Fopen((p.dfintegral_file + "/meta.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      H5LTread_dataset_int(file, "total_size", &total_size);
      H5Fclose(file);*/
      double int_mem = total_size * sizeof(std::complex<double>) / (1024. * 1024. * 1024);
      std::cerr << "Memory required for static one-electron objects per process: " << static_1e_mem << " GB"
                << std::endl;
      std::cerr << "Memory required for Coulomb integrals per node: " << shared_Coul_mem + int_mem * node_size_
                << " GB" << std::endl;
      std::cerr << "Memory required for Polarization per node: " << shared_P_mem
                << " GB" << std::endl;
      std::cerr << "Total memory required per node: "
                << 2 * gf_mem + shared_Coul_mem + shared_P_mem + (int_mem + static_1e_mem) * node_size_
                << " GB" << std::endl;
      std::cerr << "####################################################" << std::endl;
    }

    if (!world_rank_) {
      std::cerr << "*Warning*: The definition of damp parameter has been changed(1: no damp, 0: full damp)! "
                   "Please double check if you are using damping!" << std::endl;
      std::cerr << "Params:" << std::endl;
      std::cerr << p << std::endl;
    }

    if (npoly_%2 == 1 and !IR_) {
      throw std::logic_error("npoly_ has to be even number when Chebyshev grid is used!");
    }

    read_input_data(input_path_);

    if (!const_density_) nel_found_ = double(nel_);
    nw_ = ft_.wsample_fermi().size();

  } // constructor

  virtual ~sc_loop_block_t() {
    deallocate_shared_memory();
    delete sc_type_;
  }

    /*
     * ====================================================
     *                      Getters
     * ====================================================
     */

    inline double mu() const { return mu_; }

    inline size_t npoly() const { return npoly_; }

    inline size_t nts() const { return nts_; }

    inline size_t nk() const { return nk_; }

    inline size_t nw() const { return nw_; }

    inline double beta() const { return beta_; }

    inline size_t ink() const { return ink_; }

    inline size_t ns() const { return ns_; }

    inline int world_rank() const { return world_rank_; }

    inline int world_size() const { return world_size_; }

    inline solver_type_e scf_type() const { return scf_type_; }

    inline bool const_density() const { return const_density_; }

    inline double energy() const { return energy_; }

    inline double ehf() const { return ehf_; }

    inline double enuc() const { return enuc_; }

    inline double nel() const { return nel_; }

    inline double nel_found() const { return nel_found_; }

    inline double madelung() const { return madelung_; }

    inline const params_t &params() const { return params_; }

    inline tensor<dcomplex, 2> &S_k() { return S_k_; }
    inline const tensor<dcomplex, 2> &S_k() const { return S_k_; }

    inline const tensor<dcomplex, 2> &F_k() const { return F_k_; }
    inline tensor<dcomplex, 2> &F_k() { return F_k_; }

    inline const tensor<dcomplex, 2> &H_k() const { return H_k_; }
    inline tensor<dcomplex, 2> & H_k() { return H_k_; }

    inline const transformer_t &ft() const { return ft_; }

    inline const tensor_view<dcomplex, 3> &G_tau() const { return G_tau_; }

    inline const tensor<dcomplex, 2> &dmr() const { return dmr_; }

    inline tensor<dcomplex, 2> &dmr() { return dmr_; }

    inline const tensor_view<dcomplex, 3> &Selfenergy() const { return Selfenergy_; }

    inline const symmetry_utils_t &symm_utils() const { return symm_utils_; }

    inline solver_block_t &solver() const { return *solver_;}

    /* ==================================================== */

    /**
     * Construct correlated Green's function using non-uniform Matsubara grids
     */
    void init_g(bool check_convergence = false);

    template<size_t N, typename St>
    static void make_hermitian_kao(tensor_base<std::complex<double>, N, St> &X, const symmetry_utils_t &symm_utils) {
      // Dimension of the rest of arrays
      size_t dim1 = std::accumulate(X.shape().begin(), X.shape().end()-1, 1ul, std::multiplies<size_t>());
      for (size_t i = 0; i < dim1; ++i) {
        for (int ik = 0; ik < symm_utils.ink(); ++ik) {
          int n_block = symm_utils.ao_sizes_irre(ik).size();
          int kslice_offset = symm_utils.kao_slice_offsets_irre()(ik);
          const auto &offsets_k = symm_utils.ao_offsets_irre(ik);
          const auto &sizes_k = symm_utils.ao_sizes_irre(ik);
          for (int ia = 0; ia < n_block; ++ia) {
            MMatrixX<dcomplex> Xm(X.data() + i*symm_utils.flat_ao_size() + kslice_offset + offsets_k(ia),
                                  sizes_k(ia), sizes_k(ia));
            Xm = 0.5 * (Xm + Xm.conjugate().transpose().eval());
          }
        }
      }
    }

    private:
    params_t params_;
    // Brillouin zone utilities
    symmetry_utils_t symm_utils_;
    // number of polynomials
    size_t npoly_;
    // number of tau points
    size_t nts_;
    // number of frequency points
    size_t nw_;
    // number of k-points
    size_t nk_;
    // number of k-points in the reduced Brillouin zone
    size_t ink_;
    // number of spins
    size_t ns_;
    // Fock matrix
    tensor<dcomplex, 2> F_k_;
    // Overlap matrix
    tensor<dcomplex, 2> S_k_;
    // Core potential
    tensor<dcomplex, 2> H_k_;
    // Magnetic field
    double Bz_;
    // Madelung constant
    double madelung_;
    // Nuclei energy
    double enuc_;
    // One-body part of energy
    double e1e_;
    // HF energy
    double ehf_;
    // 2nd order energy
    double energy_;
    // Inverse temperature
    double beta_;
    // Chemical potential
    double mu_;
    // do constant density calculations
    bool const_density_;
    // number of electrons in the unit cell
    double nel_;
    double nel_found_;
    // tolerance
    double tol_;

    // path to inital solution in k-space
    std::string input_path_;
    // path to the results file
    std::string results_file_;
    // tau point grid
    itime_mesh_t tau_mesh_;
    // Fourier transform class
    transformer_t ft_;
    // Run HF, GW or GF2
    solver_type_e scf_type_;
    // Use IR basis or not
    bool IR_;
    // Green's function
    tensor_view<dcomplex, 3> G_tau_;  // ts_kij
    // data ptr for G_tau:
    std::complex<double> *G_tau_data_;
    // Self-energy
    tensor_view<dcomplex, 3> Selfenergy_;  // ts_kij
    std::complex<double> *Sigma_tau_data_;
    // Density matrix
    tensor<dcomplex, 2> dmr_; // s_kij

    // MPI
    MPI_Comm world_comm_;
    int world_rank_;
    int world_size_;

    // MPI for intra-node shared memory communication
    MPI_Comm node_comm_;
    int node_rank_;
    int node_size_;

    // MPI for internode communication (reduction of results
    MPI_Comm internode_comm_;
    int internode_rank_;
    int internode_size_;

    MPI_Win win_G_;
    MPI_Win win_Sigma_;

    // Many-body solver
    std::unique_ptr<solver_block_t> solver_;

    sc_type *sc_type_;

    // statistics
    execution_statistic_t statistics_;

    void initialize_shmem_communicators();

    /// allocation of Green's function
    void allocate_shared_memory();

    /// deallocation of Green's function
    void deallocate_shared_memory() {
      MPI_Win_free(&win_G_);
      MPI_Win_free(&win_Sigma_);
    }

    /// Read initial k-space solution
    void read_input_data(const std::string &path);
  };

} // namespace symmetry_mbpt

#endif //SYMMETRYMBPT_SC_LOOP_BLOCK_T_H
