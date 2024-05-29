
import numpy as np
from pyscf import gto as mgto
from pyscf.pbc import tools, gto, scf, df, dft
from pyscf.pbc.df import rsdf_builder, gdf_builder
import pyscf.lib.chkfile as chk
import argparse
import os
import h5py
import distutils.util

import common.integral_utils as int_utils
from integrals.GDF.df_rsdf_builder import BLKGDF


def wrap_k(k):

    while k < 0:
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0:
        k = k - 1
    return k


def wrap_1stBZ(k):

    while k < -0.5:
        k = k + 1
    while (k - 4.9999999999e-1) > 0.0:
        k = k - 1
    return k


def fold_back_to_1stBZ(kpts):

    for i, ik in enumerate(kpts):
        kpts[i] = np.array([wrap_1stBZ(kk) for kk in ik])
    return kpts


def find_k_in_mesh(kmesh, k):

    k1 = np.asarray(kmesh)
    Nk = k1.shape[0]
    for i in range(Nk):
        if np.allclose(k1[i], k, rtol=1.e-4, atol=1.e-5):
            return i

    #print(kmesh)
    #print(k)
    raise RuntimeError(f'can not find k in mesh. \n {k} \n {kmesh}')


def inversion_sym(kmesh_scaled):
    ind = np.arange(np.shape(kmesh_scaled)[0])
    weight = np.zeros(np.shape(kmesh_scaled)[0])
    kscaled = [[] for i in range(len(kmesh_scaled))]
    for i, ki in enumerate(kmesh_scaled):
        ki = [wrap_1stBZ(l) for l in ki]
        kscaled[i] = ki

    # Time-reversal symmetry
    Inv = (-1) * np.identity(3)
    for i, ki in enumerate(kscaled):
        ki = np.dot(Inv, ki)
        ki = [wrap_1stBZ(l) for l in ki]
        for l, kl in enumerate(kscaled[:i]):
            if np.allclose(ki, kl):
                ind[i] = l
                break

    uniq = np.unique(ind, return_counts=True)
    for i, k in enumerate(uniq[0]):
        weight[k] = uniq[1][i]
    ir_list = uniq[0]

    # Mark down time-reversal-reduced k-points
    conj_list = np.zeros(len(kscaled))
    for i, k in enumerate(ind):
        if i != k:
            conj_list[i] = 1

    return ir_list, ind, weight, conj_list


def parse_basis(basis_list):

    print(basis_list, len(basis_list) % 2)
    if len(basis_list) % 2 == 0:
        b = {}
        for atom_i in range(0, len(basis_list), 2):
            bas_i = basis_list[atom_i + 1]
            if os.path.exists(bas_i) :
                with open(bas_i) as bfile:
                    bas = mgto.parse(bfile.read())
            # if basis specified as a standard basis
            else:
                bas = bas_i
            b[basis_list[atom_i]] = bas
        return b
    else:
        return basis_list[0]


def parse_geometry(g):

    if os.path.exists(g) :
        with open(g) as gf:
            res = gf.read()
    else:
        res = g
    return res


def str2bool(v):

    return bool(distutils.util.strtobool(v))


def init_pbc_params(a, atoms):
    parser = argparse.ArgumentParser(description="GF2 initialization script")
    parser.add_argument("--a", type=parse_geometry, default=a, help="lattice geometry")
    parser.add_argument("--atom", type=parse_geometry, default=atoms, help="positions of atoms")
    parser.add_argument("--nk", type=int, default=4, help="number of k-points in each direction")
    parser.add_argument("--Nk", type=int, default=0,
                        help="number of plane-waves in each direction for integral evaluation")
    parser.add_argument("--basis", type=str, nargs="*", default=["sto-3g"],
                        help="basis sets definition. First specify atom then basis for this atom")
    parser.add_argument("--auxbasis", type=str, nargs="*", default=[None], help="auxiliary basis")
    parser.add_argument("--ecp", type=str, nargs="*", default=[None], help="effective core potentials")
    parser.add_argument("--pseudo", type=str, nargs="*", default=[None], help="pseudopotential")
    parser.add_argument("--type", type=int, default=0, help="storage type")
    parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh shift")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh center")
    parser.add_argument("--xc", type=str, nargs="*", default=[None], help="XC functional")
    parser.add_argument("--dm0", type=str, nargs=1, default=None, help="initial guess for density matrix")
    parser.add_argument("--df_int", type=int, default=1, help="prepare density fitting integrals or not")
    parser.add_argument("--orth", type=int, default=0, help="Transform to orthogonal basis or not")
    parser.add_argument("--beta", type=float, default=None,
                        help="Emperical parameter for even-Gaussian auxiliary basis")
    parser.add_argument("--active_space", type=int, nargs='+', default=None, help="active space orbitals")
    parser.add_argument("--spin", type=int, default=0, help="Local spin")
    parser.add_argument("--restricted", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default='false',
                        help="Spin restricted calculations.")

    parser.add_argument("--df_type", type=str, default='BLKGDF',
                        help="density fitting type, choose from GDF, RSGDF, RSGDF_S, BLKGDF")
    parser.add_argument("--j2c_eig_always", type=str2bool, default=True,
                        help="enforce eigenvalue decomposition for decomposing j2c")
    parser.add_argument("--linear_dep_threshold", type=float, default=1e-9,
                        help="linear dependence threshold of j2c sqrt")

    parser.add_argument("--Ci_group", type=str2bool, default=False, help="only use Ci group in symmetry analysis")
    parser.add_argument("--symmorphic_group", type=str2bool, default=False,
                        help="only use symmorphic group in symmetry analysis")
    parser.add_argument("--nonsymmorphic_group", type=str2bool, default=False,
                        help="only use nonsymmorphic group in symmetry analysis")
    parser.add_argument("--data_path", type=str, default="./", help="data path for interaction transformation test")

    parser.add_argument("--transform_file", type=str, default=None,
                        help="hdf5 results file from C++ SymmetryAdaptation code")
    parser.add_argument("--aux_transform_file", type=str, default=None,
                        help="hdf5 results file for auxiliary orbitals from C++ SymmetryAdaptation code")
    parser.add_argument("--time_reversal", type=str2bool, default=False,
                        help="apply time reversal relation in k space structure")

    args = parser.parse_args()
    args.basis = parse_basis(args.basis)
    args.auxbasis = parse_basis(args.auxbasis)
    args.ecp = parse_basis(args.ecp)
    args.pseudo = parse_basis(args.pseudo)
    args.xc = parse_basis(args.xc)
    if args.xc is not None:
        args.mean_field = dft.KRKS if args.restricted else dft.KUKS
    else:
        args.mean_field = scf.KRHF if args.restricted else scf.KUHF
    args.ns = 1 if args.restricted else 2
    return args


def cell(args):
    c = gto.M(
        a=args.a,
        atom=args.atom,
        unit='A',
        basis=args.basis,
        ecp=args.ecp,
        pseudo=args.pseudo,
        verbose=7,
        spin=args.spin,
    )
    _a = c.lattice_vectors()
    if np.linalg.det(_a) < 0:
        raise "Lattice are not in right-handed coordinate system. Please correct your lattice vectors"
    return c


# -- integrals related --

def read_dm(dm0, dm_file):

    """
    Read density matrix from smaller kmesh
    """

    nao  = dm0.shape[-1]
    nkpts = dm0.shape[1]
    dm   = np.zeros((2,nao,nao),dtype=np.complex128)
    f    = h5py.File(dm_file, 'r')
    dm[:,:,:] = f['/dm_gamma'][:]
    f.close()
    dm_kpts = np.repeat(dm[:,None, :, :], nkpts, axis=1)
    return dm_kpts


def solve_mean_field(args, mydf, mycell):

    print("Solve LDA")

    #prepare and solve DFT
    mf    = args.mean_field(mycell, mydf.kpts).density_fit()
    if args.xc is not None:
        mf.xc = args.xc
    #mf.max_memory = 10000
    mydf._cderi = "cderi.h5"
    mf.kpts = mydf.kpts
    mf.with_df = mydf
    mf.diis_space = 16
    mf.max_cycle = 100
    mf.chkfile = 'tmp.chk'
    if os.path.exists("tmp.chk"):
        init_dm = mf.from_chk('tmp.chk')
        mf.kernel(init_dm)
    elif args.dm0 is not None:
        init_dm = mf.get_init_guess()
        init_dm = read_dm(init_dm, args.dm0)
        mf.kernel(init_dm)
    else:
        mf.kernel()
    mf.analyze()
    return mf


def save_data(args, mycell, mf, kmesh, Nk, nk, NQ, F, S, T, Zs, last_ao, filename="input.h5"):

    inp_data = h5py.File(filename, "w")
    inp_data["grid/k_mesh"] = kmesh
    inp_data["grid/k_mesh_scaled"] = mycell.get_scaled_kpts(kmesh)

    # -- for UGF2 input only
    nkpts = kmesh.shape[0]
    inp_data["grid/index"] = np.arange(nkpts)
    inp_data["grid/weight"] = np.ones(nkpts)
    inp_data["grid/ink"] = nkpts
    inp_data["grid/ir_list"] = np.arange(nkpts)
    inp_data["grid/conj_list"] = np.zeros(nkpts)

    inp_data["HF/Nk"] = Nk
    inp_data["HF/nk"] = nk
    inp_data["HF/Energy"] = mf.e_tot
    inp_data["HF/Energy_nuc"] = mf.cell.energy_nuc()
    inp_data["HF/Fock-k"] = F.view(np.float64).reshape(*F.shape, 2)
    inp_data["HF/Fock-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/S-k"] = S.view(np.float64).reshape(*S.shape, 2)
    inp_data["HF/S-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/H-k"] = T.view(np.float64).reshape(*T.shape, 2)
    inp_data["HF/H-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/madelung"] = tools.pbc.madelung(mycell, kmesh)
    inp_data["HF/mo_energy"] = mf.mo_energy
    inp_data["HF/mo_coeff"] = mf.mo_coeff
    inp_data["mulliken/Zs"] = Zs
    inp_data["mulliken/last_ao"] = last_ao
    inp_data["params/nao"] = S.shape[2]
    inp_data["params/nel_cell"] = mycell.nelectron
    inp_data["params/nk"] = kmesh.shape[0]
    inp_data["params/NQ"] = NQ
    if args.active_space is not None:
        inp_data["as/indices"] = np.array(args.active_space)
    inp_data.close()
    chk.save(filename, "Cell", mycell.dumps())


def compute_df_int(args, mycell, kpts, nao, X_k):

    """
    Generate density-fitting integrals for correlated methods
    """

    if bool(args.df_int):
        if args.j2c_eig_always:
            df.rsdf_builder._RSGDFBuilder.j2c_eig_always = True
            df.gdf_builder._CCGDFBuilder.j2c_eig_always = True
        # Use gaussian density fitting to get fitted densities
        if args.df_type == 'RSGDF':
            df_method = df.RSGDF
            mydf = df.RSGDF(mycell)
            print('use RSGDF')
        elif args.df_type == 'GDF':
            df_method = df.GDF
            mydf = df.GDF(mycell)
            print('use GDF')
        elif args.df_type == 'BLKGDF':
            df_method = BLKGDF
            mydf = BLKGDF(mycell)
            mydf.linear_dep_threshold = args.linear_dep_threshold
        else:
            raise NotImplementedError
        mydf.j2c_eig_always = args.j2c_eig_always

        if args.auxbasis is not None:
            mydf.auxbasis = args.auxbasis
        elif args.beta is not None:
            mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
        # Coulomb kernel mesh
        if args.Nk > 0:
            mydf.mesh = [args.Nk, args.Nk, args.Nk]
        # Use Ewald for divergence treatment
        mydf.exxdiv = 'ewald'
        weighted_coulG_old = df_method.weighted_coulG
        df_method.weighted_coulG = int_utils.weighted_coulG_ewald
        int_utils.compute_integrals(mycell, mydf, kpts, nao, X_k, "df_int", "cderi_ewald.h5", True)

        mydf = None
        # Use gaussian density fitting to get fitted densities
        mydf = df_method(mycell)
        mydf.j2c_eig_always = args.j2c_eig_always

        if args.auxbasis is not None:
            mydf.auxbasis = args.auxbasis
        elif args.beta is not None:
            mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
        # Coulomb kernel mesh
        if args.Nk > 0:
            mydf.mesh = [args.Nk, args.Nk, args.Nk]
        df_method.weighted_coulG = weighted_coulG_old
        int_utils.compute_integrals(mycell, mydf, kpts, nao, X_k, "df_hf_int", "cderi.h5", True)

        return mydf


def compute_df_int_ewald(args, mycell, kpts, nao, X_k):

    """
    Generate density-fitting integrals for correlated methods
    """

    if bool(args.df_int):
        if args.j2c_eig_always:
            df.rsdf_builder._RSGDFBuilder.j2c_eig_always = True
            df.gdf_builder._CCGDFBuilder.j2c_eig_always = True
        # Use gaussian density fitting to get fitted densities
        if args.df_type == 'RSGDF':
            df_method = df.RSGDF
            mydf = df.RSGDF(mycell)
            print('use RSGDF')
        elif args.df_type == 'GDF':
            df_method = df.GDF
            mydf = df.GDF(mycell)
            print('use GDF')
        elif args.df_type == 'BLKGDF':
            df_method = BLKGDF
            mydf = BLKGDF(mycell)
            mydf.linear_dep_threshold = args.linear_dep_threshold
        else:
            raise NotImplementedError
        mydf.j2c_eig_always = args.j2c_eig_always

        if args.auxbasis is not None:
            mydf.auxbasis = args.auxbasis
        elif args.beta is not None:
            mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)
        # Coulomb kernel mesh
        if args.Nk > 0:
            mydf.mesh = [args.Nk, args.Nk, args.Nk]
        # Use Ewald for divergence treatment
        mydf.exxdiv = 'ewald'
        weighted_coulG_old = df_method.weighted_coulG
        df_method.weighted_coulG = int_utils.weighted_coulG_ewald
        int_utils.compute_integrals(mycell, mydf, kpts, nao, X_k, "df_int", "cderi_ewald.h5", True)
