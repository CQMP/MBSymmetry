
from common.common_utils import *
from common.blk_utils import *
from common.blk_idx_utils import *
from common.symmetry_utils import check_symmetry
from common.symmetry_utils import load_oreps, load_dirac_characters, apply_time_reversal_dirac
from transforms.block_diagonalization import block_diagonalize_matrix, flat_block_matrix
from transforms.k_space_transform import transform_all, check_little_cogroup_transform
from transforms.interaction_diagonalization import main as diag_interaction


def read_input(input_file):

    # return kmesh, F, S, T

    def complex_mat(mat):
        return mat[..., 0] + 1.j * mat[..., 1]

    f = h5py.File(input_file, 'r')
    kmesh = f["grid/k_mesh_scaled"][()]
    F = f["/HF/Fock-k"][()]
    S = f["/HF/S-k"][()]
    T = f["/HF/H-k"][()]
    madelung = f["/HF/madelung"][()]
    e_nuc = f["/HF/Energy_nuc"][()]
    e_hf = f["HF/Energy"][()]
    f.close()

    return kmesh, complex_mat(F), complex_mat(S), complex_mat(T), madelung, e_nuc, e_hf


def transform_input(args, mat_vec, k_dirac, tol=1e-6):

    res = []
    for idx, mat in enumerate(mat_vec):
        print()
        print('--------')
        print('matrix', idx)
        blk_mat_spin = []
        flat_mat_spin = []
        for s in range(args.ns):
            print('spin', s)
            blk_mat = block_diagonalize_matrix(mat[s], k_dirac, check=True, tol=tol)
            blk_mat_spin.append(np.stack(blk_mat))
            flat_mat = flat_block_matrix(blk_mat, k_dirac)
            flat_mat_spin.append(flat_mat)
            print()
        blk_mat_spin = np.stack(blk_mat_spin)
        flat_mat_spin = np.stack(flat_mat_spin)
        res.append(blk_mat_spin)
        res.append(flat_mat_spin)

    return tuple(res)


def main_matrix(args, save=True, rot=False, save_rot=True):

    kmesh_input, F, S, T, madelung, e_nuc, e_hf = read_input(args.input_file)
    kmesh_input = np.vectorize(wrap_k)(kmesh_input)
    args.ns = F.shape[0]

    k_struct, orbital_rep, aux_rep = load_oreps(args.transform_file, args.aux_transform_file, args.time_reversal)
    kmesh = np.vectorize(wrap_k)(k_struct.kmesh)

    np.testing.assert_array_almost_equal(kmesh, kmesh_input)

    print()
    print('checking unit cell ovlp matrix')
    check_symmetry(S[0], k_struct, orbital_rep)
    transform_all(S[0], orbital_rep, k_struct.transform_table, tol=1e-6)
    check_little_cogroup_transform(S[0], orbital_rep, k_struct.little_cogroup, tol=1e-6)

    # a list of dirac characters in k space
    k_dirac_orep = load_dirac_characters(args.transform_file)
    # apply time reversal relation
    k_dirac_orep = apply_time_reversal_dirac(k_dirac_orep, k_struct)

    # 4d array followed by 2d array, (s, k, nao, nao) (s, flat_mat)
    F_blk, F, S_blk, S, T_blk, T= transform_input(args, [F, S, T], k_dirac_orep, tol=1e-6)
    print_dirac_info(k_dirac_orep)
    if save:
        save_blk_data(F, S, T, madelung, e_nuc, e_hf, args.flat_input_file)
        save_blk_data(F_blk, S_blk, T_blk, madelung, e_nuc, e_hf, args.blk_input_file)

    # save symmetry related info to a separate file
    # atomic orbitals
    kao_slice_offsets, kao_slice_sizes, kao_nblocks, \
        ao_offsets, ao_sizes, ao_block_offsets = find_blk_index(k_dirac_orep)
    print('ao indices')
    print(kao_slice_offsets, kao_slice_sizes, kao_nblocks, ao_offsets, ao_sizes, ao_block_offsets, sep='\n')
    if save:
        save_blk_index(args.symmetry_file, kao_slice_offsets, kao_slice_sizes, kao_nblocks,
                       ao_offsets, ao_sizes, ao_block_offsets, prefix='ao')

    if rot:
        orep_blk = transform_orep(orbital_rep, k_dirac_orep, k_struct)
        print()
        print('checking unit cell ovlp matrix after block diagonalization')
        check_symmetry(S_blk[0], k_struct, orep_blk)
        check_little_cogroup_transform(S_blk[0], orep_blk, k_struct.little_cogroup, tol=1e-6)

        # flat_orep: list of tensors, list: k point, tensor: (op, flat_block)
        orep_flat, orep_irreps, \
            orep_offsets, orep_sizes = find_orep_blocks(orep_blk, k_dirac_orep, k_struct.transform_table)
        if save_rot:
            save_orep_blk_info(args.symmetry_rot_file, orep_blk, orep_flat, orep_irreps, orep_offsets, orep_sizes,
                               name='orep')
            k_struct.save(args.symmetry_rot_file)


def main_interaction_single(args, k_struct, kmesh, k_dirac_orep, k_dirac_auxrep, filename, foldername,
                            save=True, rot=False, save_rot=True, aux_rep=None, save_blk=True):
    k_dirac_auxrep, j2c_blk, j2c_blk_sqrt, j2c_blk_sqrt_inv, VQ_flat, VQ_blk, \
        kpair_slice_offsets, kpair_slice_sizes, kpair_nblk, \
        tensor_offsets, tensor_irreps, kptij_idx = diag_interaction(args, k_struct, k_dirac_orep,
                                                                    k_dirac_auxrep, filename, return_blk=save_blk)

    if save:
        save_VQ_meta(args.blk_integral_path + f'{foldername}/', kptij_idx, kmesh, sum([x.size for x in VQ_flat]))
        if save_blk:
            save_VQ_data(args.blk_integral_path + f'{foldername}/VQ_block_0.h5', VQ_blk)
            del VQ_blk
        VQ_flat = np.concatenate(VQ_flat)
        #save_VQ_meta(args.blk_integral_path + f'{foldername}/', kptij_idx, kmesh, VQ_flat.size)
        save_VQ_data(args.blk_integral_path + f'{foldername}/VQ_flat_0.h5', VQ_flat)
        del VQ_flat

    if rot:
        auxrep_blk = transform_orep(aux_rep, k_dirac_auxrep, k_struct)
        print()
        print('checking j2c after block diagonalization')
        check_symmetry(j2c_blk, k_struct, auxrep_blk)
        check_little_cogroup_transform(j2c_blk, auxrep_blk, k_struct.little_cogroup, tol=1e-6)
        # auxrep is only used for transforming polarization, so use cderi_ewald
        auxrep_blk_trans = transform_auxrep_blk(auxrep_blk, j2c_blk_sqrt, j2c_blk_sqrt_inv, k_struct)
        auxrep_flat, auxrep_irreps, \
            auxrep_offsets, auxrep_sizes = find_orep_blocks(auxrep_blk_trans, k_dirac_auxrep, k_struct.transform_table)
        if save_rot:
            save_orep_blk_info(args.symmetry_rot_file, auxrep_blk_trans, auxrep_flat, auxrep_irreps,
                               auxrep_offsets, auxrep_sizes, name='auxrep')

    return k_dirac_auxrep, kpair_slice_offsets, kpair_slice_sizes, kpair_nblk, tensor_offsets, tensor_irreps


def main_interaction(args, save=True, rot=False, save_rot=True):

    k_struct, orbital_rep, aux_rep = load_oreps(args.transform_file, args.aux_transform_file, args.time_reversal)
    k_dirac_orep = load_dirac_characters(args.transform_file)
    k_dirac_auxrep = load_dirac_characters(args.aux_transform_file)
    # apply time reversal relation
    k_dirac_orep = apply_time_reversal_dirac(k_dirac_orep, k_struct)

    kmesh = np.vectorize(wrap_k)(k_struct.kmesh)

    # block diagonalize interaction tensor
    k_dirac_auxrep, ewald_kpair_slice_offsets, ewald_kpair_slice_sizes, \
        ewald_kpair_nblk, ewald_tensor_offsets, ewald_tensor_irreps = \
        main_interaction_single(args, k_struct, kmesh, k_dirac_orep, k_dirac_auxrep,
                                filename='cderi_ewald_raw.h5', foldername='df_int',
                                save=save, rot=rot, save_rot=False, aux_rep=aux_rep)

    if save:
        save_kpair_blk_index(args.symmetry_file, ewald_kpair_slice_offsets,
                             ewald_kpair_slice_sizes, ewald_kpair_nblk,
                             ewald_tensor_offsets, ewald_tensor_irreps)

    # auxiliary basis
    qaux_slice_offsets, qaux_slice_sizes, qaux_nblocks, \
        aux_offsets, aux_sizes, aux_block_offsets = find_blk_index(k_dirac_auxrep)
    print('aux indices')
    print(qaux_slice_offsets, qaux_slice_sizes, qaux_nblocks, aux_offsets, aux_sizes, aux_block_offsets, sep='\n')
    if save:
        save_blk_index(args.symmetry_file, qaux_slice_offsets, qaux_slice_sizes, qaux_nblocks,
                       aux_offsets, aux_sizes, aux_block_offsets, prefix='aux')


def main():

    args = init_input_params()

    main_matrix(args, save=True, rot=True, save_rot=True)
    main_interaction(args, save=True, rot=True, save_rot=True)


if __name__ == '__main__':

    main()
