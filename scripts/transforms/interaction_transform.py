
import numpy as np
import h5py

import common.common_utils as comm
from common.symmetry_utils import check_symmetry
from integrals.GDF.df_rsdf_builder import BLKGDF


def get_j2c(filename, kmesh):

    h5swap = h5py.File(filename, 'r')
    j2c_uniq_kpts = h5swap['j2c/j2c_uniq_kpts']
    uniq_kpts = h5swap['j2c/uniq_kpts']
    scaled_uniq_kpts = h5swap['j2c/scaled_uniq_kpts']
    kpts_idx_pairs = h5swap['j2c/kpts_idx_pairs']

    j2c_full = [None for i in range(len(kmesh))]
    for j2c_idx, (k, k_conj) in enumerate(kpts_idx_pairs):
        j2c = h5swap[f'j2c/{j2c_idx}'][()]
        kpt = scaled_uniq_kpts[k]
        index = comm.find_k_in_mesh(kmesh, kpt)
        j2c_full[index] = j2c

        if k == k_conj:
            continue

        kpt = scaled_uniq_kpts[k_conj]
        index = comm.find_k_in_mesh(kmesh, kpt)
        j2c_full[index] = j2c.conj()

    h5swap.close()

    return j2c_full


def check_j2c_transform(filename, symm_utils, QOrep=None):

    j2c_full = get_j2c(filename, symm_utils.kmesh)
    print(j2c_full[0].shape)

    auxrep = symm_utils.fullk_auxrep if QOrep is None else QOrep
    check_symmetry(mat_vec=j2c_full, k_struct=symm_utils, orbital_rep=auxrep)


def get_j3c(filename, kmesh_in, cell, auxcell, full=True):
    h5swap = h5py.File(filename, 'r')
    kpts = h5swap['/kpts'][()]
    if 'kmesh' in h5swap.keys():
        kmesh = h5swap['/kmesh'][()]
    else:
        kmesh = cell.get_scaled_kpts(kpts)
    #np.testing.assert_array_almost_equal(k_struct.kmesh, np.vectorize(comm.wrap_1stBZ)(kmesh))

    nkpts = kpts.shape[0]
    # convention is kj - ki = q
    if full:
        kptij_idx = [(i, j) for i in range(nkpts) for j in range(nkpts)]
        kptij_lst = np.asarray([(ki, kj) for ki in kpts for kj in kpts])  # absolute kpts
        kptis = kptij_lst[:, 0]
        kptjs = kptij_lst[:, 1]
        kptij_scaled = np.asarray([(ki, kj) for ki in kmesh for kj in kmesh])
    else:
        kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i + 1)]
        kptij_idx = [(i, j) for i in range(kpts.shape[0]) for j in range(i + 1)]

        kptij_lst = np.asarray(kptij_lst)
        kptij_idx = np.asarray(kptij_idx)
        kptis = kptij_lst[:, 0]
        kptjs = kptij_lst[:, 1]
        kptij_scaled = np.asarray([(ki, kmesh[j]) for i, ki in enumerate(kmesh) for j in range(i + 1)])

    qpts_scaled = kptij_scaled[:, 1] - kptij_scaled[:, 0]
    qpts_scaled = comm.fold_back_to_1stBZ(qpts_scaled)
    # print(qpts_scaled)
    # exit()
    qpt_idx = []
    for q in qpts_scaled:
        qpt_idx.append(comm.find_k_in_mesh(kmesh_in, q))
    #print(qpt_idx)
    num_kpair_stored = len(kptij_idx)

    NQ, nao = auxcell.nao_nr(), cell.nao_nr()

    mydf = BLKGDF(cell)
    mydf.kpts = kmesh
    mydf._cderi = filename
    cnt = 0
    chunk_size = num_kpair_stored
    buffer = np.zeros((num_kpair_stored, NQ, nao, nao), dtype=complex)
    for i in range(num_kpair_stored):
        k1 = kptis[i]
        k2 = kptjs[i]
        # auxiliary basis index
        s1 = 0
        for XXX in mydf.sr_loop((k1, k2), max_memory=4000, compact=False):
            LpqR = XXX[0]
            LpqI = XXX[1]
            Lpq = (LpqR + LpqI * 1j).reshape(LpqR.shape[0], nao, nao)
            buffer[cnt % chunk_size, s1:s1 + Lpq.shape[0], :, :] = Lpq[0:Lpq.shape[0], :, :]
            # s1 = NQ at maximum.
            s1 += Lpq.shape[0]
        cnt += 1

    j3c_vec = buffer

    h5swap.close()

    return j3c_vec, kptij_idx, qpt_idx


def check_j3c_transform(filename, k_struct, cell, auxcell, KOrep, QOrep):

    j3c_vec, kptij_idx, qpt_idx = get_j3c(filename, k_struct.kmesh, cell, auxcell)
    print(j3c_vec.shape)

    check_q_transform(j3c_vec, k_struct, KOrep, QOrep, kptij_idx, qpt_idx)


def get_VQ(path, kmesh_in):

    # TODO: generalize this to multiple chunks
    f = h5py.File(path + 'VQ_0.h5', 'r')
    VQ = f['/0'][()]
    VQ = VQ[..., ::2] + 1.j * VQ[..., 1::2]
    f.close()

    meta = h5py.File(path + 'meta.h5', 'r')
    kpts = meta['/grid/k_mesh'][()]
    kmesh = meta['/grid/k_mesh_scaled'][()]
    k_pair_idx = meta['/grid/kpair_idx'][()]

    kptij_idx = [(i, j) for i in range(kpts.shape[0]) for j in range(i + 1)]
    kptij_scaled = np.asarray([(ki, kmesh[j]) for i, ki in enumerate(kmesh) for j in range(i + 1)])
    qpts_scaled = kptij_scaled[:, 1] - kptij_scaled[:, 0]
    qpts_scaled = comm.fold_back_to_1stBZ(qpts_scaled)
    # print(qpts_scaled)
    # exit()
    qpt_idx = []
    for q in qpts_scaled:
        qpt_idx.append(comm.find_k_in_mesh(kmesh_in, q))

    return VQ, kptij_idx, qpt_idx


def check_q_transform(j3c_vec, k_struct, KOrep, QOrep, kptij_idx, qpt_idx):

    num_kpair_stored = len(kptij_idx)
    # check transformation
    for idx in range(num_kpair_stored):
        j3c = j3c_vec[idx]
        ki_idx, kj_idx = kptij_idx[idx]
        q_idx = qpt_idx[idx]

        print()
        print("target momenta:", ki_idx, kj_idx, q_idx)

        # suppose we have q in the irreducible wedge
        q_star_idx = np.where(k_struct.stars_rep_idx == k_struct.ir_list[q_idx])[0][0]  # this is the star q is in
        q_star = k_struct.stars_idx[q_star_idx]
        q_star_rep = k_struct.stars_rep_idx[q_star_idx]
        q_star_ops = k_struct.stars_ops[q_star_idx]  # list of list

        # need to find the operation corresponding to q_rep -> q
        q_ops = q_star_ops[q_star.index(q_idx)]  # a list of operations

        # this is for testing purpose only. In practice, probably we only need to one operation
        for q_op in q_ops:
            print('operation:', q_op)

            # given q_op, find k_j' with O(k_j') = k_j
            kj_prime_idx = k_struct.transform_table_rev[kj_idx, q_op]
            ki_prime_idx = k_struct.transform_table_rev[ki_idx, q_op]

            print("source momenta:", ki_prime_idx, kj_prime_idx, q_star_rep)

            # get all the transformation matrix
            orep_ki = KOrep[ki_prime_idx, q_op]
            orep_kj = KOrep[kj_prime_idx, q_op]
            orep_q = QOrep[q_star_rep, q_op]

            if ki_prime_idx >= kj_prime_idx:
                source_kpair = np.array([ki_prime_idx, kj_prime_idx])
                source_kpair_idx = np.where(np.all(kptij_idx == source_kpair, axis=1))
                j3c_rep = j3c_vec[source_kpair_idx][0]
            else:
                source_kpair = np.array([kj_prime_idx, ki_prime_idx])
                source_kpair_idx = np.where(np.all(kptij_idx == source_kpair, axis=1))
                j3c_rep = j3c_vec[source_kpair_idx][0]
                for Q in range(QOrep.shape[-1]):
                    j3c_rep[Q] = j3c_rep[Q].conj().T

            j3c_trans = np.einsum("PQ,ab,Qbc,cd->Pad", orep_q, orep_ki, j3c_rep, orep_kj.conj().T, optimize="greedy")

            print('diff norm:', np.linalg.norm(j3c-j3c_trans))
            if np.linalg.norm(j3c-j3c_trans) > 1e-3:
                print('Warning: transformation fails')
