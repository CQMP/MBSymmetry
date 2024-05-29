
from pyscf.pbc import gto
from pyscf.df import addons
import pyscf.lib.chkfile as chk

from common.blk_utils import *
from common.symmetry_utils import read_j2c, load_oreps, load_dirac_characters
from common.common_utils import find_k_in_mesh, wrap_k
from integrals.GDF.df_rsdf_builder_ref import BLKRSGDFBuilder
from transforms.interaction_transform import get_j3c, get_VQ
from transforms.block_diagonalization import block_diagonalize_matrix


CDERI_FOLDER = {'cderi_raw.h5': 'df_hf_int/',
                'cderi_ewald_raw.h5': 'df_int/'}
CDERI_FILE = {'cderi_raw.h5': 'cderi.h5',
              'cderi_ewald_raw.h5': 'cderi_ewald.h5'}


def make_cells(filename, auxbasis):

    cell_bytes = chk.load(filename, "/Cell")
    cell = gto.Cell()
    cell = cell.loads(cell_bytes)
    cell.build()

    auxcell = addons.make_auxmol(cell, auxbasis)

    return cell, auxcell


def enforce_time_reversal(k_dirac, q_conj_index):

    for q in range(len(q_conj_index)):
        cq = q_conj_index[q]
        if q != cq:
            k_dirac[q].U = k_dirac[cq].U.conj()
            k_dirac[q].block_size = k_dirac[cq].block_size
            k_dirac[q].block_start_idx = k_dirac[cq].block_start_idx
            k_dirac[q].block_idx = k_dirac[cq].block_idx
    return k_dirac


def block_diagonalize_j3c(j3c_vec, k_dirac_orep, k_dirac_auxrep, kptij_idx, qpt_idx, q_conj_index):

    print('block diagonalize j3c')

    num_kpair_stored = len(kptij_idx)
    blk_j3c_vec = []
    for idx in range(num_kpair_stored):
        j3c = j3c_vec[idx]
        ki_idx, kj_idx = kptij_idx[idx]
        q_idx = qpt_idx[idx]

        print("momenta:", ki_idx, kj_idx, q_idx)

        if q_conj_index[q_idx] == q_idx:
            U_q = k_dirac_auxrep[q_idx].U
        else:
            U_q = k_dirac_auxrep[q_conj_index[q_idx]].U.conj()

        U_i = k_dirac_orep[ki_idx].U
        U_j = k_dirac_orep[kj_idx].U

        j3c_blk = np.einsum("PQ,ab,Qbc,cd->Pad", U_q.conj().T, U_i.conj().T, j3c, U_j, optimize="greedy")
        j3c_blk[np.abs(j3c_blk) < 1e-6] = 0.
        blk_j3c_vec.append(j3c_blk)

    return blk_j3c_vec


def sqrt_block_j2c(j2c_blk, dirac, df_builder):

    # sqrt of a single j2c

    size = j2c_blk.shape[0]
    n_block = dirac.block_start_idx.shape[0]

    st = dirac.block_start_idx
    en = np.append(dirac.block_start_idx[1:], [size])

    j2c_sqrt_out = np.zeros_like(j2c_blk)
    j2c_sqrt_inv_out = np.zeros_like(j2c_blk)
    for j in range(n_block):
        tmp = j2c_blk[st[j]:en[j], st[j]:en[j]]
        cd_j2c = df_builder.decompose_j2c(tmp)
        j2c_blk_sqrt_inv, _, _, j2c_blk_sqrt = cd_j2c

        neffi = j2c_blk_sqrt.shape[1]
        neffj = j2c_blk_sqrt_inv.shape[0]
        j2c_sqrt_out[st[j]:en[j], st[j]:(st[j]+neffi)] = j2c_blk_sqrt
        j2c_sqrt_inv_out[st[j]:(st[j]+neffj), st[j]:en[j]] = j2c_blk_sqrt_inv

    return j2c_sqrt_inv_out, j2c_sqrt_out


def get_block_sqrt_j2c(path, cderi_name, mycell, auxcell, kmesh, k_dirac_auxrep):

    kpts = mycell.get_abs_kpts(kmesh)
    df_builder = BLKRSGDFBuilder(mycell, auxcell, kpts)
    df_builder.j2c_eig_always = True

    filename = path + cderi_name
    h5swap = h5py.File(filename, 'r')
    j2c_uniq_kpts = h5swap['j2c/j2c_uniq_kpts'][()]
    uniq_kpts = h5swap['j2c/uniq_kpts'][()]
    scaled_uniq_kpts = h5swap['/j2c/scaled_uniq_kpts'][()]
    kpts_idx_pairs = h5swap['j2c/kpts_idx_pairs'][()]

    j2c_blk_full = [None for i in range(len(kmesh))]
    j2c_blk_sqrt_full, j2c_blk_sqrt_inv_full = [None for i in range(len(kmesh))], [None for i in range(len(kmesh))]
    k_conj_index = [None for i in range(len(kmesh))]
    for j2c_idx, (k, k_conj) in enumerate(kpts_idx_pairs):
        j2c = h5swap[f'j2c/{j2c_idx}'][()]
        kpt = scaled_uniq_kpts[k]
        index = find_k_in_mesh(kmesh, kpt)

        j2c_blk = block_diagonalize_matrix([j2c], k_dirac_auxrep[index:index+1])[0]
        j2c_sqrt_inv, j2c_sqrt = sqrt_block_j2c(j2c_blk, k_dirac_auxrep[index], df_builder)

        [j2c_blk, j2c_sqrt_inv, j2c_sqrt] = \
            filter_block_matrix([j2c_blk, j2c_sqrt_inv, j2c_sqrt], [k_dirac_auxrep[index]]*3)

        j2c_blk_full[index], j2c_blk_sqrt_full[index], j2c_blk_sqrt_inv_full[index] = j2c_blk, j2c_sqrt, j2c_sqrt_inv
        k_conj_index[index] = index

        if k_conj is None or k == k_conj:
            continue

        kpt = scaled_uniq_kpts[k_conj]
        conj_index = find_k_in_mesh(kmesh, kpt)
        j2c_blk_full[conj_index], j2c_blk_sqrt_full[conj_index], j2c_blk_sqrt_inv_full[conj_index] \
            = j2c_blk.conj(), j2c_sqrt.conj(), j2c_sqrt_inv.conj()
        k_conj_index[conj_index] = index
    h5swap.close()

    return j2c_blk_full, j2c_blk_sqrt_inv_full, j2c_blk_sqrt_full, k_conj_index


def compute_VQ(j2c_sqrt_inv_vec, j3c_vec, kptij_idx, qpt_idx):

    num_kpair_stored = len(kptij_idx)
    VQ_vec = []
    for idx in range(num_kpair_stored):
        j3c = j3c_vec[idx]
        q_idx = qpt_idx[idx]
        j2c_sqrt_inv = j2c_sqrt_inv_vec[q_idx]
        VQ = np.tensordot(j2c_sqrt_inv, j3c, axes=(1, 0))
        VQ_vec.append(VQ)

    return np.stack(VQ_vec, axis=0)


def check_VQ_full(path, cderi_name, mycell, auxcell, kmesh):

    j3c, kptij_idx, qpt_idx = get_j3c(path + cderi_name, kmesh, mycell, auxcell, full=True)
    VQ, kptij_idx_V, qpt_idx_V = get_j3c(path + CDERI_FILE[cderi_name], kmesh, mycell, auxcell, full=True)
    np.testing.assert_array_almost_equal(kptij_idx, kptij_idx_V)
    np.testing.assert_array_almost_equal(qpt_idx, qpt_idx_V)

    j2c, j2c_sqrt, j2c_sqrt_inv, _ = read_j2c(np.vectorize(wrap_k)(kmesh), filename=path+cderi_name)
    VQ_compute = compute_VQ(j2c_sqrt_inv, j3c, kptij_idx, qpt_idx)
    np.testing.assert_array_almost_equal(VQ, VQ_compute)


def main(args, k_struct, k_dirac_orep, k_dirac_auxrep, cderi_name, return_blk=False):

    path = args.integral_path

    mycell, auxcell = make_cells(path+'input_symm.h5', args.auxbasis)

    kmesh = np.vectorize(wrap_k)(k_struct.kmesh)
    kpts = mycell.get_abs_kpts(kmesh)

    if args.check_VQ:
        check_VQ_full(path, cderi_name, mycell, auxcell, k_struct.kmesh)

    # this gives full list with size nk
    j2c_blk, j2c_blk_sqrt_inv, j2c_blk_sqrt, q_conj_index \
        = get_block_sqrt_j2c(path, cderi_name, mycell, auxcell, kmesh, k_dirac_auxrep)
    k_dirac_auxrep = enforce_time_reversal(k_dirac_auxrep, q_conj_index)

    # TODO: full j3c and VQ might be too large to store in memory.
    j3c, kptij_idx, qpt_idx = get_j3c(path + cderi_name, k_struct.kmesh, mycell, auxcell)
    j3c_blk = block_diagonalize_j3c(j3c, k_dirac_orep, k_dirac_auxrep, kptij_idx, qpt_idx, q_conj_index)
    del j3c
    VQ_blk = compute_VQ(j2c_blk_sqrt_inv, j3c_blk, kptij_idx, qpt_idx)
    del j3c_blk

    # flat VQ
    VQ_flat, kpair_slice_offsets, kpair_slice_sizes, kpair_nblk, \
        tensor_offsets, tensor_irreps = find_VQ_blocks(VQ_blk,
                                                       kptij_idx,
                                                       qpt_idx,
                                                       k_dirac_orep,
                                                       k_dirac_auxrep)
    if not return_blk:
        del VQ_blk
        VQ_blk = None

    return k_dirac_auxrep, j2c_blk, j2c_blk_sqrt, j2c_blk_sqrt_inv, VQ_flat, VQ_blk, \
        kpair_slice_offsets, kpair_slice_sizes, kpair_nblk, tensor_offsets, tensor_irreps, kptij_idx
