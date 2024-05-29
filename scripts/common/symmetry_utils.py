
import h5py
import numpy as np

from transforms.k_space_transform import transform_matrix
from transforms.symmetrization import k_space_symmetrization
from transforms.k_space_structure import KSpaceStructure
from transforms.dirac_character import DiracCharacter

import common.common_utils as comm


def save_symmetry_info(k_struct, orbital_rep, aux_rep, auxrep_trans=None, filename="input.h5"):

    inp_data = h5py.File(filename, "a")
    if "/symmetry" in inp_data:
        del inp_data["symmetry"]
    symm_group = inp_data.require_group("symmetry")

    # TODO: will update this later to store the transformation matrix that gives the matrix at all k points
    orep = orbital_rep
    symm_group["KspaceORep"] = orep.view(np.float64).reshape(*orep.shape, 2)
    symm_group["KspaceORep"].attrs["__complex__"] = np.int8(1)
    if aux_rep is not None:
        symm_group["KspaceAuxRep"] = aux_rep.view(np.float64).reshape(*aux_rep.shape, 2)
        symm_group["KspaceAuxRep"].attrs["__complex__"] = np.int8(1)
    if auxrep_trans is not None:
        symm_group["KspaceAuxRepTrans"] = auxrep_trans.view(np.float64).reshape(*auxrep_trans.shape, 2)
        symm_group["KspaceAuxRepTrans"].attrs["__complex__"] = np.int8(1)

    k_struct.save(inp_data, close=False)
    inp_data.close()


def save_identity_symmetry_info(nk, nao, naux, filename="input.h5"):

    inp_data = h5py.File(filename, "a")
    symm_group = inp_data.require_group("symmetry")

    orep = np.zeros((nk, 1, nao, nao), dtype=complex)
    auxrep = np.zeros((nk, 1, naux, naux), dtype=complex)
    for i in range(nk):
        orep[i, 0] = np.eye(nao, dtype=complex)
        auxrep[i, 0] = np.eye(naux, dtype=complex)

        symm_group[f"Kpoint/{i}/ops"] = np.array([0])
        symm_group[f"Star/{i}/k_idx"] = np.array([i])

    symm_group["n_star"] = nk
    symm_group["KspaceORep"] = orep.view(np.float64).reshape(*orep.shape, 2)
    symm_group["KspaceORep"].attrs["__complex__"] = np.int8(1)
    #symm_group["KspaceAuxRep"] = auxrep.view(np.float64).reshape(*auxrep.shape, 2)
    #symm_group["KspaceAuxRep"].attrs["__complex__"] = np.int8(1)
    symm_group["KspaceAuxRepTrans"] = auxrep.view(np.float64).reshape(*auxrep.shape, 2)
    symm_group["KspaceAuxRepTrans"].attrs["__complex__"] = np.int8(1)

    symm_group["KStruct/k_ibz_index"] = np.arange(nk)
    symm_group["KStruct/irr_list"] = np.arange(nk)
    symm_group["KStruct/weight"] = np.ones(nk)

    inp_data.close()


def check_symmetry(mat_vec, k_struct, orbital_rep, tol=1e-4):

    transform_matrix(mat_vec,
                     stars=k_struct.stars_idx,
                     star_reps=k_struct.stars_rep_idx,
                     star_ops=k_struct.stars_ops,
                     KOrep=orbital_rep,
                     k_mesh=k_struct.kmesh,
                     tol=tol)


def symmetrize(mat_vec, k_struct, orbital_rep):

    return k_space_symmetrization(mat_vec,
                                  stars=k_struct.stars_idx,
                                  star_reps=k_struct.stars_rep_idx,
                                  star_ops=k_struct.stars_ops,
                                  KOrep=orbital_rep,
                                  k_mesh=k_struct.kmesh)


def transform_auxrep(auxrep, k_struct: KSpaceStructure, path='./'):

    kmesh = np.vectorize(comm.wrap_k)(k_struct.kmesh)

    # (naux, ntrunc), (ntrunc, naux)
    j2c, j2c_sqrt, j2c_sqrt_inv, _ = read_j2c(kmesh, path+'cderi_ewald_raw.h5')

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

    for i in range(nk):
        j2c_ori = j2c[i]
        j2c_s, j2c_s_i = j2c_sqrt[i], j2c_sqrt_inv[i]
        j2c_trans = j2c_s @ j2c_s_i @ j2c_ori
        np.testing.assert_array_almost_equal(j2c_ori, j2c_trans)

    print('j2c check')
    check_symmetry(j2c, k_struct, auxrep)

    return auxrep_trans


def read_j2c(kmesh, filename='cderi.h5'):

    h5swap = h5py.File(filename, 'r')
    j2c_uniq_kpts = h5swap['j2c/j2c_uniq_kpts'][()]
    uniq_kpts = h5swap['j2c/uniq_kpts'][()]
    scaled_uniq_kpts = h5swap['/j2c/scaled_uniq_kpts'][()]
    kpts_idx_pairs = h5swap['j2c/kpts_idx_pairs'][()]

    j2c_full = [None for i in range(len(kmesh))]
    j2c_sqrt_full, j2c_sqrt_inv_full = [None for i in range(len(kmesh))], [None for i in range(len(kmesh))]
    k_conj_pairs = []
    for j2c_idx, (k, k_conj) in enumerate(kpts_idx_pairs):
        j2c = h5swap[f'j2c/{j2c_idx}'][()]
        j2c_sqrt = h5swap[f'j2c/j2c_sqrt/{j2c_idx}'][()]
        j2c_sqrt_inv = h5swap[f'j2c/j2c_sqrt_inv/{j2c_idx}'][()]
        kpt = scaled_uniq_kpts[k]
        index = comm.find_k_in_mesh(kmesh, kpt)
        j2c_full[index], j2c_sqrt_full[index], j2c_sqrt_inv_full[index] = j2c, j2c_sqrt, j2c_sqrt_inv

        if k_conj is None or k == k_conj:
            k_conj_pairs.append((index, index))
            continue

        kpt = scaled_uniq_kpts[k_conj]
        conj_index = comm.find_k_in_mesh(kmesh, kpt)
        j2c_full[conj_index], j2c_sqrt_full[conj_index], j2c_sqrt_inv_full[conj_index] \
            = j2c.conj(), j2c_sqrt.conj(), j2c_sqrt_inv.conj()
        k_conj_pairs.append((index, conj_index))

    h5swap.close()

    return j2c_full, j2c_sqrt_full, j2c_sqrt_inv_full, k_conj_pairs


# ----------------------------------------------------------------------------------------------------------------------
def complex_mat(mat):

    return mat[..., 0] + 1.j * mat[..., 1]


def load_oreps(filename, bosonic_filename=None, time_reversal=True):

    k_struct = KSpaceStructure.load(filename, time_reversal=time_reversal)

    f = h5py.File(filename, 'r')
    orbital_rep = complex_mat(f['/KspaceORep'][()])
    f.close()

    if bosonic_filename is not None:
        f = h5py.File(bosonic_filename, 'r')
        aux_rep = complex_mat(f['/KspaceORep'][()])
        f.close()
    else:
        aux_rep = None

    return k_struct, orbital_rep, aux_rep


def load_dirac_characters(filename):

    if filename is None:
        return None

    # return a list of length nk that contains Dirac character of each k point
    f = h5py.File(filename, 'r')
    nk = f['/grid/k_mesh'][()].shape[0]
    k_dirac_characters = []
    for i in range(nk):
        dirac = DiracCharacter()
        U = f[f'/Kpoint/{i}/Dirac_character/U'][()]
        dirac.U = U[:, :, 0] + 1.j * U[:, :, 1]
        dirac.block_size = f[f'/Kpoint/{i}/Dirac_character/block_size'][()]
        dirac.block_start_idx = f[f'/Kpoint/{i}/Dirac_character/block_start_idx'][()]
        dirac.block_idx = np.append(dirac.block_start_idx, [dirac.U.shape[-1]])
        k_dirac_characters.append(dirac)

    f.close()
    return k_dirac_characters


def apply_time_reversal_dirac(k_dirac, k_struct):

    if not k_struct.time_reversal: return k_dirac

    for k in range(k_struct.nk):
        ck = k_struct.conj_index[k]
        if k != ck:
            k_dirac[k].U = k_dirac[ck].U.conj()
            k_dirac[k].block_size = k_dirac[ck].block_size
            k_dirac[k].block_start_idx = k_dirac[ck].block_start_idx
            k_dirac[k].block_idx = k_dirac[ck].block_idx
    return k_dirac
