
import argparse
import copy

import h5py
import numpy as np
import os

from common.common_utils import str2bool, parse_basis


def init_input_params():
    parser = argparse.ArgumentParser(description="block diagonalization script")

    parser.add_argument("--input_file", type=str, default="input_symm.h5",
                        help="hdf5 file with input integrals")
    parser.add_argument("--blk_input_file", type=str, default="input_blk.h5",
                        help="hdf5 file to store block integrals")
    parser.add_argument("--flat_input_file", type=str, default="input_flat.h5",
                        help="hdf5 file to store flat integrals")

    parser.add_argument("--symmetry_file", type=str, default="symmetry_info.h5",
                        help="hdf5 file to store block information")
    parser.add_argument("--symmetry_rot_file", type=str, default="symmetry_rot.h5",
                        help="hdf5 file to store rotation matrices with block structure")

    parser.add_argument("--transform_file", type=str, default=None,
                        help="hdf5 results file from C++ SymmetryAdaptation code")
    parser.add_argument("--aux_transform_file", type=str, default=None,
                        help="hdf5 results file for auxiliary orbitals from C++ SymmetryAdaptation code")
    parser.add_argument("--block_diag", type=str2bool, default=True,
                        help="preform block diagonalization of matrices in orbital space")
    parser.add_argument("--block_diag_interaction", type=str2bool, default=True,
                        help="preform block diagonalization of decomposed interaction")

    parser.add_argument("--integral_path", type=str, default='./', help="path with integral files")
    parser.add_argument("--blk_integral_path", type=str, default='./block/', help="path with block integral files")
    parser.add_argument("--auxbasis", type=str, nargs="*", default=[None], help="auxiliary basis")
    parser.add_argument("--check_VQ", type=str2bool, default=False,
                        help="cross check VQ can be correctly computed")
    parser.add_argument("--check_U", type=str2bool, default=False,
                        help="cross check Uijkl can be correctly computed")
    parser.add_argument("--time_reversal", type=str2bool, default=False,
                        help="apply time reversal relation in k space structure")

    args = parser.parse_args()
    args.auxbasis = parse_basis(args.auxbasis)

    return args


def save_blk_data(F, S, T, madelung, e_nuc, e_hf, filename, group_name='/'):

    inp_data = h5py.File(filename, "a")
    blk_group = inp_data.require_group(group_name)
    blk_group["HF/Fock-k"] = F.view(np.float64).reshape(*F.shape, 2)
    blk_group["HF/Fock-k"].attrs["__complex__"] = np.int8(1)
    blk_group["HF/S-k"] = S.view(np.float64).reshape(*S.shape, 2)
    blk_group["HF/S-k"].attrs["__complex__"] = np.int8(1)
    blk_group["HF/H-k"] = T.view(np.float64).reshape(*T.shape, 2)
    blk_group["HF/H-k"].attrs["__complex__"] = np.int8(1)
    blk_group["HF/madelung"] = madelung
    blk_group["/HF/Energy_nuc"] = e_nuc
    blk_group["/HF/Energy"] = e_hf
    inp_data.close()


def save_VQ_meta(filename, kptij_idx, kmesh, tot_size):

    if not os.path.exists(filename):
        os.mkdir(filename)

    num_kpair_stored = len(kptij_idx)
    # -- fake input for UGF2 integrals
    kij_conj = np.arange(num_kpair_stored)
    kij_trans = np.arange(num_kpair_stored)
    kpair_irre_list = np.argwhere(kij_conj == kij_trans)[:, 0]
    num_kpair_stored = len(kpair_irre_list)

    data = h5py.File(filename + 'meta.h5', 'a')
    data["chunk_size"] = num_kpair_stored
    data["chunk_indices"] = [0]
    data["total_size"] = tot_size
    data["grid/kpair_idx"] = kptij_idx
    data["grid/num_kpair_stored"] = num_kpair_stored
    data["grid/k_mesh_scaled"] = kmesh
    # for UGF2 integrals
    data["grid/conj_pairs_list"] = kij_conj
    data["grid/trans_pairs_list"] = kij_trans
    data["grid/kpair_irre_list"] = kpair_irre_list
    data.close()


def save_VQ_data(filename, VQ):

    data = h5py.File(filename, 'a')
    data['/0'] = VQ.view(np.float64)
    data.close()


def print_dirac_info(k_dirac):

    for idx, dirac in enumerate(k_dirac):
        print('block index of k point', idx)
        print(np.append(dirac.block_start_idx, [dirac.block_start_idx[-1]+dirac.block_size[-1]]))


def transform_orep(orep, k_dirac, k_struct):

    nk, nop, nao, _ = orep.shape
    orep_trans = np.empty_like(orep, dtype=complex)

    trans_table = k_struct.transform_table
    for i in range(nk):
        for op in range(nop):
            mat = orep[i, op]
            target = trans_table[i, op]  # op transform i to target
            orep_trans[i, op] = k_dirac[target].U.conj().T @ mat @ k_dirac[i].U

    return orep_trans


def find_orep_blocks(orep, k_dirac, transform_table, tol=1e-5):

    nk, nop, _, _ = orep.shape

    np.set_printoptions(precision=3)

    trans_table = transform_table
    # oreps of different k points might have blocks of different sizes,
    # but total size would be the same for k points in the same star
    flat_oreps = []
    flat_blk_irreps = []
    flat_irreps_sizes = []
    flat_irreps_offsets = []
    for i in range(nk):
        print('k point', i)
        dirac_i = k_dirac[i]
        st_i = dirac_i.block_idx[:-1]
        en_i = dirac_i.block_idx[1:]
        print(st_i, en_i)
        k_oreps = []
        k_irreps = []
        k_irreps_sizes = []
        k_irreps_offsets = []
        for op in range(nop):
            j = trans_table[i, op]   # target
            print('target k point', j)
            dirac_j = k_dirac[j]
            st_j = dirac_j.block_idx[:-1]
            en_j = dirac_j.block_idx[1:]
            print(st_j, en_j)
            assert len(st_i) == len(st_j)

            mat = orep[i, op]
            flat_mat = []
            flat_irreps = []
            flat_offsets = [0]
            flat_sizes = []
            # transformed oreps should have shape (irrep_j, irrep_i)
            n_blocks = 0
            for r in range(len(st_j)):
                for c in range(len(st_i)):
                    blk = mat[st_j[r]:en_j[r], st_i[c]:en_i[c]]
                    #print(blk)
                    #blk = mat[st_i[c]:en_i[c], st_j[r]:en_j[r]]
                    if np.max(np.abs(blk)) > tol:
                        n_blocks += 1
                        flat_mat.append(blk.reshape(-1))
                        flat_irreps.append(np.array([r, c]))
                        flat_sizes.append(blk.shape)
                        flat_offsets.append(flat_offsets[-1]+blk.size)
            print('number of total blocks', len(st_j)*len(st_i))
            print('number of nonzero blocks', n_blocks)
            flat_mat = np.concatenate(flat_mat)
            flat_irreps = np.stack(flat_irreps)
            k_oreps.append(flat_mat)
            k_irreps.append(flat_irreps)
            k_irreps_offsets.append(flat_offsets[:-1])
            k_irreps_sizes.append(flat_sizes)

        k_oreps = np.stack(k_oreps)
        k_irreps = np.stack(k_irreps)
        k_irreps_offsets = np.stack(k_irreps_offsets)
        k_irreps_sizes = np.stack(k_irreps_sizes)
        flat_oreps.append(k_oreps)
        flat_blk_irreps.append(k_irreps)
        flat_irreps_offsets.append(k_irreps_offsets)
        flat_irreps_sizes.append(k_irreps_sizes)

    return flat_oreps, flat_blk_irreps, flat_irreps_offsets, flat_irreps_sizes


def check_little_cogroup_orep_blocks(orep, k_dirac, little_cogroup, tol=1e-5):

    nk, _, nao, _ = orep.shape
    for i in range(nk):
        print()
        print('k point', i)
        ops = little_cogroup[i]
        print('little cogroup', ops)
        dirac_i = k_dirac[i]
        st_i = dirac_i.block_start_idx
        en_i = np.append(dirac_i.block_start_idx[1:], [nao])
        print(st_i, en_i)
        for op in ops:
            print('operation', op)
            mat = orep[i, op]
            flat_mat = []
            flat_irreps = []
            flat_idx = [0]
            n_blocks = 0
            for r in range(len(st_i)):
                for c in range(len(st_i)):
                    blk = mat[st_i[r]:en_i[r], st_i[c]:en_i[c]]
                    # print(blk)
                    # blk = mat[st_i[c]:en_i[c], st_j[r]:en_j[r]]
                    if np.linalg.norm(blk) > tol:
                        n_blocks += 1
                        flat_mat.append(blk.reshape(-1))
                        flat_irreps.append(np.array([r, c]))
                        flat_idx.append(flat_idx[-1] + blk.size)
            print('number of nonzero blocks', n_blocks)
            print('nonzero irreps', flat_irreps)


def find_VQ_blocks(VQ, kptij_idx, qpt_idx, k_dirac_orep, k_dirac_auxrep, tol=1e-6):

    print()
    print('find VQ blocks')

    num_kpair_stored = len(kptij_idx)
    flat_VQ = []
    kpair_slice_offsets = []
    kpair_slice_sizes = []
    kpair_nblk = []
    tensor_offsets = []  # -> list of list
    tensor_irreps = []  # -> list of list

    k_offsets = 0
    for idx in range(num_kpair_stored):
        ki_idx, kj_idx = kptij_idx[idx]
        q_idx = qpt_idx[idx]
        dirac_i, dirac_j = k_dirac_orep[ki_idx], k_dirac_orep[kj_idx]
        dirac_q = k_dirac_auxrep[q_idx]
        print("momenta:", ki_idx, kj_idx, q_idx)
        idx_i = dirac_i.block_idx
        idx_j = dirac_j.block_idx
        idx_q = dirac_q.block_idx

        j3c = VQ[idx]
        flat_mat = []
        flat_irreps = []
        flat_offset = [0]
        n_blocks = 0
        for i in range(len(idx_i)-1):
            for j in range(len(idx_j)-1):
                for q in range(len(idx_q)-1):
                    blk = j3c[idx_q[q]:idx_q[q+1], idx_i[i]:idx_i[i+1], idx_j[j]:idx_j[j+1]]
                    #if np.linalg.norm(blk) > tol:
                    if np.max(np.abs(blk)) > tol:
                        n_blocks += 1
                        flat_mat.append(blk.reshape(-1))
                        flat_irreps.append(np.array([q, i, j]))
                        flat_offset.append(flat_offset[-1]+blk.size)
        print('numer of total blocks', (len(idx_i)-1) * (len(idx_j)-1) * (len(idx_q)-1))
        print('number of nonzero blocks', n_blocks)
        kpair_nblk.append(n_blocks)
        flat_mat = np.concatenate(flat_mat)
        flat_irreps = np.stack(flat_irreps)

        flat_VQ.append(flat_mat)
        tensor_irreps.append(flat_irreps)
        tensor_offsets.append(flat_offset[:-1])

        kpair_slice_offsets.append(k_offsets)
        slice_size = flat_offset[-1]
        kpair_slice_sizes.append(slice_size)
        k_offsets += slice_size
    #flat_VQ = np.concatenate(flat_VQ)

    return flat_VQ, kpair_slice_offsets, kpair_slice_sizes, kpair_nblk, tensor_offsets, tensor_irreps


def transform_auxrep_blk(auxrep, j2c_sqrt, j2c_sqrt_inv, k_struct):

    trans_table = k_struct.transform_table
    nk = k_struct.nk
    order = k_struct.group.order
    auxrep_trans = np.zeros_like(auxrep, dtype=complex)
    for i in range(nk):
        for op in range(order):
            j = trans_table[i, op]
            neffj = j2c_sqrt_inv[j].shape[0]
            neffi = j2c_sqrt[i].shape[1]
            auxrep_trans[i, op][:neffj, :neffi] = j2c_sqrt_inv[j] @ auxrep[i, op] @ j2c_sqrt[i]

    return auxrep_trans


def filter_block_matrix(k_mat, k_dirac):

    nmat = len(k_mat)
    for i in range(nmat):
        size = k_mat[i].shape[0]
        n_block = k_dirac[i].block_start_idx.shape[0]

        st = k_dirac[i].block_start_idx
        en = np.append(k_dirac[i].block_start_idx[1:], [size])
        mat = copy.deepcopy(k_mat[i])
        k_mat[i] = np.zeros_like(mat)
        for n in range(n_block):
            k_mat[i][st[n]:en[n], st[n]:en[n]] = mat[st[n]:en[n], st[n]:en[n]]
        if np.linalg.norm(mat - k_mat[i]) > 1e-5:
            raise RuntimeError('filtering non block matrix')

    return k_mat
