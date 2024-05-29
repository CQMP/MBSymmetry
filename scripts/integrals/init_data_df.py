
from pyscf.df import addons
from pyscf.pbc.df import rsdf_builder, gdf_builder

from common.common_utils import *
from common.symmetry_utils import (save_symmetry_info, check_symmetry, save_identity_symmetry_info,
                                   transform_auxrep, symmetrize, load_oreps)
from integrals.GDF.df_rsdf_builder import BLKGDF, BLKRSGDFBuilder


def generate_input(args, mycell, kpts):

    """
    Generate integrals for mean-field calculations
    """

    # number of k-points in each direction for Coulomb integrals
    nk = args.nk
    # number of k-points in each direction to evaluate Coulomb kernel
    Nk = args.Nk

    if args.df_type == 'RSGDF':
        mydf = df.RSGDF(mycell)
        print('use RSGDF')
    elif args.df_type == 'GDF':
        mydf = df.GDF(mycell)
        print('use GDF')
    elif args.df_type == "BLKGDF":
        mydf = BLKGDF(mycell)
        mydf.linear_dep_threshold = args.linear_dep_threshold
    else:
        raise NotImplementedError

    if args.auxbasis is not None:
        mydf.auxbasis = args.auxbasis
    elif args.beta is not None:
        mydf.auxbasis = df.aug_etb(mycell, beta=args.beta)

    # Coulomb kernel mesh
    if Nk > 0:
        mydf.mesh = [Nk, Nk, Nk]
    mydf.kpts = kpts

    if os.path.exists("cderi.h5"):
        mydf._cderi = "cderi.h5"
    else:
        mydf._cderi_to_save = "cderi.h5"
        mydf.build()
    auxcell = addons.make_auxmol(mycell, mydf.auxbasis)
    NQ = auxcell.nao_nr()

    print('solving mean field', flush=True)
    mf = solve_mean_field(args, mydf, mycell)

    # Get Overlap and Fock matrices
    hf_dm = mf.make_rdm1().astype(dtype=np.complex128)
    S = mf.get_ovlp().astype(dtype=np.complex128)
    T = mf.get_hcore().astype(dtype=np.complex128)
    if args.xc is not None:
        vhf = mf.get_veff().astype(dtype=np.complex128)
    else:
        vhf = mf.get_veff(hf_dm).astype(dtype=np.complex128)
    print('computing HF solution', flush=True)
    F = mf.get_fock(T, S, vhf, hf_dm).astype(dtype=np.complex128)

    if len(F.shape) == 3:
        F = F.reshape((1,) + F.shape)
        hf_dm = hf_dm.reshape((1,) + hf_dm.shape)
    S = np.array((S,) * args.ns)
    T = np.array((T,) * args.ns)

    return mf, Nk, nk, NQ, F, S, T, hf_dm


def symmetrize_input(args, mat_vec, k_struct, orep):

    res = []
    for idx, mat in enumerate(mat_vec):
        print()
        print('--------')
        print('matrix', idx)
        symm_mat = np.empty_like(mat, dtype=np.complex128)
        for s in range(args.ns):
            print('spin', s)
            symm_mat[s] = symmetrize(mat[s], k_struct, orep)
        res.append(symm_mat)
    
    return tuple(res)


def check_input_symmetry(args, mat_vec, k_struct, orep):

    for idx, mat in enumerate(mat_vec):
        print()
        print('--------')
        print('matrix', idx)
        for s in range(args.ns):
            print('spin', s)
            check_symmetry(mat[s], k_struct, orep, tol=3e-3)


def main():

    # Default geometry
    a = '''2.03275,    2.03275,    0.0
           0.0,    2.03275,    2.03275
           2.03275,    0.0,    2.03275'''
    atoms = '''H   0.0   0.0    0.0
               Li 2.03275 2.03275 2.03275'''

    args = init_pbc_params(a, atoms)

    if args.j2c_eig_always:
        df.rsdf_builder._RSGDFBuilder.j2c_eig_always = True
        df.gdf_builder._CCGDFBuilder.j2c_eig_always = True

    BLKRSGDFBuilder.linear_dep_threshold = args.linear_dep_threshold

    mycell = cell(args)
    auxcell = addons.make_auxmol(mycell, args.auxbasis)

    # -- print cell info
    nao = mycell.nao_nr()
    Zs = np.asarray(mycell.atom_charges())
    print("Number of atoms: ", Zs.shape[0])
    print("Effective nuclear charge of each atom: ", Zs)
    atoms_info = np.asarray(mycell.aoslice_by_atom())
    last_ao = atoms_info[:, 3]
    print("aoslice_by_atom = ", atoms_info)
    print("Last AO index for each atom = ", last_ao)

    # -- print basis info
    print("atomic basis:")
    ao_labels = []
    for label in mycell.ao_labels():
        # print("{}{} - {}".format(label[1],label[0],label[2]))
        ao_labels.append(label)
    ao_labels = np.asarray(ao_labels)
    print(ao_labels)

    print("auxiliary basis:")
    ao_labels = []
    for label in auxcell.ao_labels():
        # print("{}{} - {}".format(label[1],label[0],label[2]))
        ao_labels.append(label)
    ao_labels = np.asarray(ao_labels)
    print(ao_labels)

    print('loading symmetry information from input files')
    k_struct, orbital_rep, aux_rep = load_oreps(args.transform_file, args.aux_transform_file)

    kmesh = np.vectorize(wrap_k)(k_struct.kmesh)
    kpts = mycell.get_abs_kpts(kmesh)
    np.testing.assert_array_almost_equal(kpts[0], np.array([0, 0, 0]))  # the first k point need to be gamma point

    # -- check symmetry with ovlp matrix before proceed
    print()
    print('checking unit cell ovlp matrix')
    S = np.asarray(mycell.pbc_intor('int1e_ovlp', kpts=kpts)) * np.complex128(1)
    check_symmetry(S, k_struct, orbital_rep)

    print()
    print('checking auxcell ovelp matrix')
    S_aux = np.asarray(auxcell.pbc_intor('int1e_ovlp', kpts=kpts)) * np.complex128(1)
    check_symmetry(S_aux, k_struct, aux_rep)

    # -- get integrals
    mf, Nk, nk, NQ, F, S, T, hf_dm = generate_input(args, mycell, kpts)
    print()
    print('check input symmetry')
    check_input_symmetry(args, [F, S, T], k_struct, orbital_rep)

    print()
    print('symmetrize input')
    F, S, T = symmetrize_input(args, [F, S, T], k_struct, orbital_rep)

    save_data(args, mycell, mf, kpts, Nk, nk, NQ, F, S, T, Zs, last_ao, 'input_symm.h5')

    save_data(args, mycell, mf, kpts, Nk, nk, NQ, F, S, T, Zs, last_ao, 'input.h5')
    save_identity_symmetry_info(k_struct.nk, nao, auxcell.nao_nr(), 'input.h5')

    compute_df_int_ewald(args, mycell, kpts, nao, X_k=None)

    aux_rep_trans = transform_auxrep(aux_rep, k_struct) if args.df_type == 'BLKGDF' else None
    save_symmetry_info(k_struct, orbital_rep, None, aux_rep_trans, filename='input_symm.h5')


if __name__ == '__main__':

    main()
