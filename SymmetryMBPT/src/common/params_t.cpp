/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include "params_t.h"

void symmetry_mbpt::params_t::init() {
  alps::hdf5::archive ar(input_file, "r");
  alps::mpi::communicator comm;
  if (!ar.is_group("params")) {
    if (!comm.rank()) {
      std::cerr << "####################################################" << std::endl;
      std::cerr << "Achtung!!! Achtung!!! Parameters group in input file is not defined. " <<
                "You are probably using old version of input " <<
                "file which is not supported by current version of this code. Enjoy!" << std::endl;
      std::cerr << "####################################################" << std::endl;
    }
    return;
  }
  ar["params/nel_cell"] >> nel_cell;
  ar["params/nao"] >> nao;
  ar["params/nk"] >> nk;
  ar["params/NQ"] >> NQ;
}

void symmetry_mbpt::params_t::save(alps::hdf5::archive &ar) const {
  ar["nel_cell"] << nel_cell;
  ar["nao"] << nao;
  ar["nk"] << nk;
  ar["NQ"] << NQ;
  ar["mu"] << mu;
  ar["beta"] << beta;
  ar["ni"] << ni;
  ar["single_prec"] << single_prec;
  ar["tol"] << tol;
  ar["CONST_DENSITY"] << CONST_DENSITY;
  ar["input_file"] << input_file;
  ar["Results"] << results_file;
  ar["dfintegral_file"] << dfintegral_file;
  ar["HF_dfintegral_file"] << dfintegral_hf_file;
  ar["TNC"] << tnc_f_file;
  ar["TNC_B"] << tnc_b_file;
  ar["symmetry_file"] << symmetry_file;
  ar["symmetry_rot_file"] << symmetry_rot_file;
  ar["rotate"] << rotate;
  ar["time_reversal"] << time_reversal;
}

symmetry_mbpt::params_t symmetry_mbpt::parse_command_line(int argc, char **argv) {

  std::unordered_map<std::string, solver_type_e> solver_map = {{"GW", GW},
                                                               {"cuGW", cuGW},
                                                               {"HF", HF},
                                                               {"cuHF", cuHF}};

  std::unordered_map<std::string, self_consistency_type_e> self_consistency_map = {{"Dyson", Dyson}};

  args::ArgumentParser parser("This is the Michigan Many-body perturbation solver for solids.");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::Positional<std::string> inifile(parser, "inifile", "The parameter file");
  auto nel_cell = define<double>(parser, "nel_cell", "number of electrons per cell");
  auto nao = define<size_t>(parser, "nao", 0, "number of atomic orbitals per cell");
  auto ns = define<size_t>(parser, "ns", "number of spins");
  auto nk = define<size_t>(parser, "nk", "Number of k-points");
  auto ink = define<size_t>(parser, "ink", "Number of reduced k-points after time-reversal symmetry");
  auto NQ = define<size_t>(parser, "NQ", 0, "Number of auxiliary basis");
  auto beta = define<double>(parser, "beta", "Inverse temperature");
  auto Bz = define<double>(parser, "Bz", 0.0, "Magnetic field in z-direction.");
  auto ni = define<size_t>(parser, "ni", "Number of chebyshev polynomials");
  auto nt_batch = define<size_t>(parser, "nt_batch", 1, "Size of tau batch in cuda GW solver");
  //auto ntauspinprocs = define<int>(parser, "ntauspinprocs", 1, "Number of processes for sub-communicator on tau and spin axes");
  auto IR = define<bool>(parser, "IR", false, "Use IR basis or not");
  auto itermax = define<int>(parser, "itermax", "Maximum number of GF2 iterations");
  auto hf_gpu = define<bool>(parser, "hf_gpu", "Whether use gpu for hf calculation");
  auto single_prec = define<int>(parser, "single_prec", 0, "Activates single and mixed precision: 0 for double precision; 1 for single precision; 2 for single precision + cleanup in double precision");
  auto sc_type = define<self_consistency_type_e>(parser, "sc_type", Dyson, "Self-consistency type: Dyson - simple Dyson equation, DCA - DCA self-consistency.", self_consistency_map);
  auto tol = define<double>(parser, "tol", 1e-9, "Double precision tolerance");

  auto mu = define<double>(parser, "mu", "Chemical potential");
  auto delta = define<double>(parser, "delta", 0.5, "Chemical potential delta");
  auto CONST_DENSITY = define<bool>(parser, "CONST_DENSITY", true, "Fix number of electrons");
  auto rotate = define<bool>(parser, "rotate", false, "Use rotation symmetry");
  auto time_reversal = define<bool>(parser, "time_reversal", false, "Use time reversal symmetry");
  /* -- */
  auto results_file = define<std::string>(parser, "Results", "sim.h5", "File to store results");
  auto input_file = define<std::string>(parser, "input_file", "input.h5", "Input file with Hartree-Fock solution and system parameters");
  auto symmetry_file = define<std::string>(parser, "symmetry_file", "symmetry_info.h5", "Input file with symmetry information");
  auto symmetry_rot_file = define<std::string>(parser, "symmetry_rot_file", "symmetry_rot.h5", "Input file with rotation matrices");
  //auto cuda_low_cpu_memory = define<bool>(parser, "cuda_low_cpu_memory", false, "True: Low CPU memory but slow, False: vice versa.");
  //auto cuda_low_gpu_memory = define<bool>(parser, "cuda_low_gpu_memory", false, "False: High memory requiremnt but fast , True: Low memory requirement but slow");
  auto dfintegral_file = define<std::string>(parser, "dfintegral_file", "df_int", "Density fitted Coulomb integrals hdf5 file");
  auto dfintegral_hf_file = define<std::string>(parser, "HF_dfintegral_file", "df_hf_int", "Density fitted Coulomb integrals for HF");
  auto tnc_f_file = define<std::string>(parser, "TNL", "TNL.h5", "Fermionic Chebyshev convolutin matrix (to be deleted");
  auto tnc_b_file = define<std::string>(parser, "TNL_B", "TNL_B.h5", "Bosonic Chebyshev convolution matrix");
  auto renorm = define<bool>(parser, "RENORM", true, "Renormalize k weights");
  auto mulliken = define<bool>(parser, "mulliken", false, "Mulliken analysis");
  auto scf_type = define<solver_type_e>(parser, "scf_type", "Type of the solver: HF, cuHF, GW, cuGW, GF2, cuGF2, GF2-direct or Ladder", solver_map);
  auto programs = define<std::string>(parser, "programs", std::vector<std::string>{{"SC"}}, "Programs to run: SC - self-consistency loop; GP - compute grand potential; TPDM - compute two-particle density matrices.");

  try{
    parser.ParseCLI(argc, argv);
  } catch (const args::Help & h) {
    std::cout<<parser;
    throw h;
  }

  INI::File ft;
  ft.Load(inifile.Get(), true);

  params_t params{detail::extract_value(ft, nel_cell),
                  detail::extract_value(ft, nao),
                  detail::extract_value(ft, ns),
                  detail::extract_value(ft, nk),
                  detail::extract_value(ft, ink),
                  detail::extract_value(ft, NQ),
                  detail::extract_value(ft, beta),
                  detail::extract_value(ft, mu),
                  detail::extract_value(ft, Bz),
                  detail::extract_value(ft, tol),
                  detail::extract_value(ft, CONST_DENSITY),
                  detail::extract_value(ft, scf_type),
                  detail::extract_value(ft, sc_type),
                  detail::extract_value(ft, nt_batch),
                  detail::extract_value(ft, single_prec),
                  detail::extract_value(ft, ni),
                  detail::extract_value(ft, IR),
                  detail::extract_value(ft, rotate),
                  detail::extract_value(ft, time_reversal),
                  detail::extract_value(ft, input_file),
                  detail::extract_value(ft, results_file),
                  detail::extract_value(ft, dfintegral_file),
                  detail::extract_value(ft, dfintegral_hf_file),
                  detail::extract_value(ft, tnc_f_file),
                  detail::extract_value(ft, tnc_b_file),
                  detail::extract_value(ft, symmetry_file),
                  detail::extract_value(ft, symmetry_rot_file),
                  detail::extract_value(ft, programs)
  };

  params.init();
  return params;
}

symmetry_mbpt::params_t symmetry_mbpt::default_parameters() {
  params_t params{0, 0, 0, 0, // nel_cell, nao, ns, nk
                  0, 0, 100, 0.0, // ink, NQ, beta, mu
                  0.0, // Bz
                  1e-9, // tol
                  true, // CONST_DENSITY
                  GW, Dyson, // scf_type, sc_type
                  1, 0, // nt_batch, single_prec
                  0, true, // ni, IR
                  false, false, // rotate, time_reversal
                  "input.h5", "sim.h5",
                  "df_int", "df_int", "TNC.h5", "TNC.h5",
                  "symmetry_info.h5",
                  "symmetry_rot.h5",
                  std::vector<std::string>{{"SC"}}};

  return params;
}
