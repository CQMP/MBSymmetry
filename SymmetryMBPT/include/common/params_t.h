#ifndef SYMMETRYMBPT_PARAMS_T_H
#define SYMMETRYMBPT_PARAMS_T_H

/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <alps/utilities/mpi.hpp>
#include <alps/hdf5.hpp>
#include <iostream>
#include <args/args.hxx>
#include <ini/iniparser.h>
#include <typeinfo>
#include "common.h"
#include "type.h"

namespace symmetry_mbpt {

  struct params_t {
    /*
     * system related
     */
    double nel_cell;
    size_t nao;
    size_t ns;
    size_t nk;
    size_t ink;
    size_t NQ;
    double beta;
    double mu;
    // magnetic field
    double Bz;

    /*
    * scf related
    */
    double tol;
    bool CONST_DENSITY;
    solver_type_e scf_type;
    self_consistency_type_e sc_type;

    /*
     * GPU related
     */
    size_t nt_batch;
    int single_prec;

    /*
     * time and frequency grid related
     */
    size_t ni; // number of grid points
    bool IR; // whether to use IR grid

    /*
     * symmetry related
     */
    bool rotate;
    bool time_reversal;

    std::string input_file;
    std::string results_file;
    std::string dfintegral_file;
    std::string dfintegral_hf_file;
    std::string tnc_f_file;
    std::string tnc_b_file;
    std::string symmetry_file;
    std::string symmetry_rot_file;

    std::vector<std::string> programs;

    void init();
    void save(alps::hdf5::archive &ar) const;
  };


  template<typename T>
  inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    //Write data
    for (const T &val: v) {
      os << val << " ";
    }
    return os;
  }

  inline std::ostream &operator<<(std::ostream &out, const symmetry_mbpt::params_t &p) {
    out << "The following major parameters are used: \n";
    out << "SCF type: " << (p.scf_type == HF || p.scf_type == cuHF ? "HF" :
                            (p.scf_type == GW || p.scf_type == cuGW ? "GW" : "Unknown")) << "\n"
        << "Cuda solver: " << (p.scf_type == cuHF || p.scf_type == cuGW ? "Yes" : "No") << "\n"
        //<< "X2C1e-Coulomb Hamiltonian: " << (p.X2C ? "True" : "False") << "\n"
        << "Number of orbitals in cell: " << p.nao << "\n"
        << "Number of auxiliary basis in cell: " << p.NQ << "\n"
        << "Number of spins: " << p.ns << "\n"
        << "Number of electrons per cell: " << p.nel_cell << "\n"
        << "Number of k-points: " << p.nk << "\n"
        << "Number of intermediate basis coefficients: " << p.ni << "\n"
        << "Inverse temperature: " << p.beta << "\n"
        << "Input file: " << p.input_file << "\n"
        << "Coulomb integral file: " << p.dfintegral_file << "\n"
        << "Coulomb integral file for HF: " << p.dfintegral_hf_file << "\n"
        << "Fermionic fourier coefficients file: " << p.tnc_f_file << "\n"
        << "Bosonic fourier coefficients file: " << p.tnc_b_file << "\n"
        << "Output file: " << p.results_file << std::endl
        << std::endl
        << "use of IR coefficients: " << p.IR << " (true: " << true << " false " << false << ")" << std::endl
        << "Symmetry file: " << p.symmetry_file << std::endl
        << "Symmetry rotation file: " << p.symmetry_rot_file << std::endl
        << "Use rotation symmetry: " << p.rotate << " (true: " << true << " false " << false << ")" << std::endl
        << "Use time reversal symmetry: " << p.time_reversal << " (true: " << true << " false " << false << ")" << std::endl
        << std::endl;

    return out;
  }

  namespace detail {
    template<typename T>
    T extract_value(const INI::File &f, args::ValueFlag <T> &parameter) {
      return parameter ? parameter.Get() : f.GetValue(parameter.Name(), parameter.GetDefault()).template Get<T>();
    };

    template<typename E>
    E extract_value(const INI::File &f, args::MapFlag <std::string, E> &parameter) {
      if (parameter) {
        return parameter.Get();
      }
      std::string value = f.GetValue(parameter.Name(), "").AsString();
      return value == "" ? parameter.GetDefault() : parameter.GetMap().count(value) > 0 ? parameter.GetMap().find(
          value)->second : throw std::logic_error("No value in enum map");
    };

    template<typename T>
    std::vector<T> extract_value(const INI::File &f, args::ValueFlagList <T> &parameter) {
      if (parameter) {
        return args::get(parameter);
      }
      std::vector<T> values;
      INI::Array array = f.GetValue(parameter.Name()).AsArray();
      if (array.Size() == 0) {
        return parameter.GetDefault();
      }
      for (int i = 0; i < array.Size(); ++i) {
        values.push_back(array[i].template AsT<T>());
      }
      return values;
    }
  } // namespace detail

  template<typename T>
  inline args::ValueFlagList <T>
  define(args::ArgumentParser &parser, const std::string &name, const std::vector<T> &def, const std::string &descr) {
    args::ValueFlagList<T> arg(parser, name, descr, {name}, def);
    return arg;
  }

  template<typename T>
  inline args::ValueFlag <T>
  define(args::ArgumentParser &parser, const std::string &name, const T &def, const std::string &descr) {
    args::ValueFlag<T> arg(parser, name, descr, {name}, def);
    return arg;
  }

  template<typename T>
  inline args::ValueFlag <T> define(args::ArgumentParser &parser, const std::string &name, const std::string &descr) {
    args::ValueFlag<T> arg(parser, name, descr, {name});
    return arg;
  }

  template<typename E>
  inline args::MapFlag <std::string, E>
  define(args::ArgumentParser &parser, const std::string &name, const E &def, const std::string &descr,
         const std::unordered_map<std::string, E> &mapper) {
    args::MapFlag<std::string, E> arg(parser, name, descr, {name}, mapper, def);
    return arg;
  }

  template<typename E>
  inline args::MapFlag <std::string, E>
  define(args::ArgumentParser &parser, const std::string &name, const std::string &descr,
         const std::unordered_map<std::string, E> &mapper) {
    args::MapFlag<std::string, E> arg(parser, name, descr, {name}, mapper);
    return arg;
  }

  params_t parse_command_line(int argc, char **argv);

  params_t default_parameters();

} // namespace symmetry_mbpt


#endif //SYMMETRYMBPT_PARAMS_T_H
