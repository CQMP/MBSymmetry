
import numpy as np
from pyscf.pbc import gto, df, tools
from pyscf.df import addons
import h5py
import os
import shutil
from numba import jit


def weighted_coulG_ewald(mydf, kpt, exx, mesh, omega=None):
    return df.aft.weighted_coulG(mydf, kpt, True, mesh, omega)


# compute partitioning
def compute_partitioning(tot_size, num_kpair_stored):
    # We have a guess for each fitted density upper bound of 150M
    ubound = 700 * 1024 * 1024
    if tot_size > ubound:
        mult = tot_size // ubound
        return num_kpair_stored // (mult + 1)
    return num_kpair_stored


def compute_integrals(mycell, mydf, kpts, nao, X_k=None, basename="df_int", cderi_name="cderi.h5", keep=True):

    kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i + 1)]
    kptij_idx = [(i, j) for i in range(kpts.shape[0]) for j in range(i + 1)]

    #kptij_lst = [(ki, kj) for ki in kpts for kj in kpts]
    #kptij_idx = [(i, j) for i in range(mydf.kpts.shape[0]) for j in range(mydf.kpts.shape[0])]
    kptij_lst = np.asarray(kptij_lst)
    kptij_idx = np.asarray(kptij_idx)
    kptis = kptij_lst[:, 0]
    kptjs = kptij_lst[:, 1]
    num_kpair_stored = len(kptis)
    print("number of k-pairs: ", num_kpair_stored)

    # -- fake input for UGF2 integrals
    kij_conj = np.arange(len(kptis))
    kij_trans = np.arange(len(kptis))
    kpair_irre_list = np.argwhere(kij_conj == kij_trans)[:,0]
    num_kpair_stored = len(kpair_irre_list)
    print("number of reduced k-pairs: ", num_kpair_stored)

    mydf.kpts = kpts
    filename = basename + "/meta.h5"
    os.system("sync")  # This is needed to synchronize the NFS between nodes
    if os.path.exists(basename):
        unsafe_rm = "rm  " + basename + "/*"
        os.system(unsafe_rm)
        # This is needed to ensure that files are removed both on the computational and head nodes:
        os.system("sync")
    os.system("mkdir -p " + basename)  # Here "-p" is important and is needed if the if condition is triggered
    if os.path.exists(cderi_name) and keep:
        mydf._cderi = cderi_name
    else:
        mydf._cderi_to_save = cderi_name
        mydf.build()

    auxcell = addons.make_auxmol(mycell, mydf.auxbasis)
    NQ = auxcell.nao_nr()
    print("NQ = ", NQ)

    single_rho_size = nao ** 2 * NQ * 16
    full_rho_size = (num_kpair_stored * single_rho_size)
    chunk_size = compute_partitioning(full_rho_size, num_kpair_stored)
    print("The chunk size: ", chunk_size)

    # open file to write integrals in
    if os.path.exists(filename):
        os.remove(filename)
    data = h5py.File(filename, "w")

    # Loop over k-point pair
    # processed densities count
    cnt = 0
    # densities buffer
    buffer = np.zeros((chunk_size, NQ, nao, nao), dtype=complex)
    Lpq_mo = np.zeros((NQ, nao, nao), dtype=complex)
    chunk_indices = []
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

        # if reach chunk size: (cnt-chunk_size) equals to chunk id.
        if cnt % chunk_size == 0:
            chunk_name = basename + "/VQ_{}.h5".format(cnt - chunk_size)
            if os.path.exists(chunk_name):
                os.remove(chunk_name)
            VQ = h5py.File(chunk_name, "w")
            VQ["{}".format(cnt - chunk_size)] = buffer.view(np.float64)
            VQ.close()
            chunk_indices.append(cnt - chunk_size)
            buffer[:] = 0.0
        # Deal the rest
    if cnt % chunk_size != 0:
        last_chunk = (num_kpair_stored // chunk_size) * chunk_size
        chunk_name = basename + "/VQ_{}.h5".format(last_chunk)
        if os.path.exists(chunk_name):
            os.remove(chunk_name)
        VQ = h5py.File(chunk_name, "w")
        VQ["{}".format(last_chunk)] = buffer.view(np.float64)
        chunk_indices.append(last_chunk)
        VQ.close()
        buffer[:] = 0.0

    data["chunk_size"] = chunk_size
    data["chunk_indices"] = np.array(chunk_indices)
    data["grid/kpair_idx"] = kptij_idx
    data["grid/num_kpair_stored"] = num_kpair_stored
    data["grid/k_mesh"] = kpts
    data["grid/k_mesh_scaled"] = mycell.get_scaled_kpts(kpts)

    # for UGF2 integrals
    data["grid/conj_pairs_list"] = kij_conj
    data["grid/trans_pairs_list"] = kij_trans
    data["grid/kpair_irre_list"] = kpair_irre_list

    data.close()
    print("Integrals have been computed and stored into {}".format(filename))
