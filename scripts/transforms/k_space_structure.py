
import numpy as np
import h5py

from transforms.space_group import SpaceGroup


class KSpaceStructure(object):

    def __init__(self, space_group: SpaceGroup, kpts: np.array):

        self.kpts = kpts  # each k point should be a row
        self.group = space_group

    def save(self, h5file, close=True):

        if isinstance(h5file, str):
            f = h5py.File(h5file, 'a')
        elif isinstance(h5file, h5py.File):
            f = h5file
        else:
            raise RuntimeError('can not recognize h5file for saving k space structure')

        symm_group = f.require_group("symmetry")

        symm_group["KStruct/k_ibz"] = self.stars_rep
        symm_group["KStruct/k_ibz_index"] = self.stars_rep_idx
        symm_group["KStruct/irr_list"] = self.ir_list
        symm_group["KStruct/irre_index"] = self.irre_index
        symm_group["KStruct/weight"] = self.weight
        symm_group["KStruct/transform_table"] = self.transform_table
        symm_group["KStruct/transform_table_rev"] = self.transform_table_rev
        symm_group["n_star"] = self.n_star

        # k, k_conj relations
        symm_group["KStruct/stars_conj_index"] = self.stars_conj_index
        symm_group["KStruct/conj_list"] = self.conj_list
        symm_group["KStruct/conj_index"] = self.conj_index
        symm_group["KStruct/conj_check"] = self.conj_check
        symm_group["KStruct/irre_conj_list"] = self.irre_conj_list
        symm_group["KStruct/irre_conj_index"] = self.irre_conj_index
        symm_group["KStruct/irre_conj_weight"] = self.irre_conj_weight
        symm_group["n_conj_star"] = self.n_conj_star
        symm_group["time_reversal"] = int(self.time_reversal)

        nk = self.nk
        for i in range(nk):
            symm_group[f"Kpoint/{i}/ops"] = self.ks_ops[i]
            symm_group[f"Kpoint/{i}/little_cogroup"] = self.little_cogroup[i]

        n_star = self.n_star
        for i in range(n_star):
            symm_group[f"Star/{i}/k_points"] = self.stars[i]
            symm_group[f"Star/{i}/k_idx"] = self.stars_idx[i]
            star_ops = self.stars_ops[i]
            for j, ops in enumerate(star_ops):
                symm_group[f"Star/{i}/operations/k_{j}"] = ops
        if close:
            f.close()

    @staticmethod
    def load(filename, time_reversal=True):

        # load both k struct info (including k mesh) and group info
        group = SpaceGroup()
        group.load(filename)

        f = h5py.File(filename, 'r')
        kpts = f['/grid/k_mesh'][()]

        k_struct = KSpaceStructure(space_group=group, kpts=kpts, compute=False)
        k_struct.nk = kpts.shape[0]
        k_struct.kmesh = f['/grid/k_mesh_scaled'][()]

        if 'symmetry' in f.keys():
            symm_group = f["symmetry"]
        else:
            symm_group = f
        k_struct.stars_rep = symm_group["KStruct/k_ibz"][()]
        k_struct.stars_rep_idx = symm_group["KStruct/k_ibz_index"][()]
        k_struct.ir_list = symm_group["KStruct/irr_list"][()]
        k_struct.irre_index = symm_group["KStruct/irre_index"][()]
        k_struct.weight = symm_group["KStruct/weight"][()]
        k_struct.transform_table = symm_group["KStruct/transform_table"][()]
        k_struct.transform_table_rev = symm_group["KStruct/transform_table_rev"][()]

        ks_ops = []
        little_cogroup = []
        for i in range(k_struct.nk):
            ks_ops.append(list(symm_group[f"Kpoint/{i}/ops"][()]))
            little_cogroup.append(list(symm_group[f"Kpoint/{i}/little_cogroup"][()]))
        k_struct.ks_ops = ks_ops
        k_struct.little_cogroup = little_cogroup

        if "n_star" in symm_group.keys():
            k_struct.n_star = symm_group["n_star"][()]
        else:
            k_struct.n_star = symm_group["Star/n_star"][()]
        stars = []
        stars_idx = []
        stars_ops = []
        for i in range(k_struct.n_star):
            stars.append(symm_group[f"Star/{i}/k_points"][()])
            stars_idx.append(list(symm_group[f"Star/{i}/k_idx"][()]))
            star_ops = []
            for j in range(len(stars_idx[-1])):
                star_ops.append(list(symm_group[f"Star/{i}/operations/k_{j}"][()]))
            stars_ops.append(star_ops)

        k_struct.stars = stars
        k_struct.stars_idx = stars_idx
        k_struct.stars_ops = stars_ops

        if time_reversal:
            k_struct.stars_conj_index = symm_group["KStruct/stars_conj_index"][()]
            k_struct.conj_list = symm_group["KStruct/conj_list"][()]
            k_struct.conj_index = symm_group["KStruct/conj_index"][()]
            k_struct.conj_check = symm_group["KStruct/conj_check"][()]
            k_struct.irre_conj_list = symm_group["KStruct/irre_conj_list"][()]
            k_struct.irre_conj_index = symm_group["KStruct/irre_conj_index"][()]
            k_struct.irre_conj_weight = symm_group["KStruct/irre_conj_weight"][()]
            k_struct.n_conj_star = len(k_struct.irre_conj_list)
            k_struct.time_reversal = symm_group["time_reversal"][()]
        else:
            k_struct.stars_conj_index = np.arange(k_struct.n_star, dtype=int)
            k_struct.conj_index = np.arange(k_struct.nk, dtype=int)
            k_struct.conj_list = np.unique(k_struct.conj_index, return_counts=False)
            k_struct.conj_check = np.zeros(k_struct.nk)
            k_struct.irre_conj_list = k_struct.stars_rep_idx[list(set(k_struct.stars_conj_index))]
            k_struct.n_conj_star = len(k_struct.irre_conj_list)
            k_struct.irre_conj_index = []
            k_struct.irre_conj_weight = np.zeros_like(k_struct.irre_conj_list)
            for i in range(k_struct.nk):
                i_conj = k_struct.conj_index[i]
                ik = k_struct.ir_list[i_conj]
                conj_idx = np.where(k_struct.irre_conj_list == ik)[0][0]
                k_struct.irre_conj_index.append(conj_idx)
                k_struct.irre_conj_weight[conj_idx] += 1
            k_struct.time_reversal = False

        f.close()

        return k_struct
