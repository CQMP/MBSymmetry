from typing import Dict

import h5py
import numpy as np

import spglib

import pyscf.pbc.gto as gto
from pyscf.pbc.symm.pyscf_spglib import cell_to_spgcell

BOHR2A = 0.529177  # 1 bohr = 0.529177 angstrom


def get_symmetry_info(cell: gto.cell.Cell, tol=1e-6) -> Dict:
    dataset = spglib.get_symmetry_dataset(cell_to_spgcell(cell), symprec=tol)  # , angle_tolerance=-1.0, hall_number=0)

    return dataset


class SpaceGroup(object):

    def __init__(self):

        self.number = 0
        self.order = 0
        self.pointgroup = None

        self.equiv_atoms = None
        self.translations = None
        self.space_rep = None
        self.reciprocal_space_rep = None

        self.spg_translations = None
        self.spg_rotations = None

        self.symmorphic = True
        self.symmorphic_indices = None
        self.nonsymmorphic_indices = None

    def load(self, filename):

        f = h5py.File(filename, 'r')
        group = f['SpaceGroup']
        self.number = group['spacegroup_number'][()]
        self.order = group['n_operations'][()]
        self.equiv_atoms = group['equiv_atoms'][()]
        self.translations = group['translations'][()]
        self.space_rep = group['space_rep'][()]
        self.reciprocal_space_rep = group['reciprocal_space_rep'][()]
        self.symmorphic = group['symmorphic'][()]
        f.close()

    def get_space_group_info(self, cell: gto.cell.Cell, tol=1e-6):

        dataset = get_symmetry_info(cell, tol)

        A = cell.lattice_vectors().transpose() * BOHR2A  # pyscf gives [a vectors] in rows, transform to columns
        A_inv = np.linalg.inv(A)

        self.number = dataset['number']
        self.pointgroup = dataset['pointgroup']

        self.equiv_atoms = dataset['equivalent_atoms']
        self.translations = np.einsum('ab,ib->ia', A, dataset['translations'], optimize='greedy')
        self.space_rep = np.einsum('ab,ibc,cd->iad', A, dataset['rotations'], A_inv, optimize='greedy')

        self.spg_translations = dataset['translations']
        self.spg_rotations = dataset['rotations']

        self.order = self.translations.shape[0]
        recip_rep = []
        for i in range(self.order):
            recip_rep.append(np.linalg.inv(self.space_rep[i]).T)
        self.reciprocal_space_rep = np.stack(recip_rep, axis=0)

        if np.max(np.abs(self.translations)) > tol:
            self.symmorphic = False

        self.symmorphic_indices = np.arange(self.order) if self.symmorphic else self.find_symmorphic_subgroup(tol)
        self.nonsymmorphic_indices = np.array(list(set(np.arange(self.order)) - set(self.symmorphic_indices)))

    def set_Ci_group(self, tol=1e-6):

        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.]])
        indices = []
        for i in range(self.order):
            if (np.linalg.norm(self.space_rep[i] - identity) < tol) \
                    or (np.linalg.norm(self.space_rep[i] + identity)) < tol:
                indices.append(i)
        if len(indices) is not 2:
            raise RuntimeError("do not have Ci subgroup")

        self.set_subgroup(indices, tol)

    def set_subgroup(self, indices, tol=1e-6):

        self.space_rep = self.space_rep[indices]
        self.reciprocal_space_rep = self.reciprocal_space_rep[indices]
        self.translations = self.translations[indices]
        self.order = len(indices)
        self.symmorphic = False if np.max(np.abs(self.translations)) > tol else True

    def find_symmorphic_subgroup(self, tol=1e-6):

        indices = []
        for i in range(self.order):
            if np.max(np.abs(self.translations[i])) < tol:
                indices.append(i)
        return np.array(indices)

    def set_symmorphic_subgroup(self, tol=1e-6):

        if self.symmorphic_indices is None:
            self.symmorphic_indices = self.find_symmorphic_subgroup(tol)

        self.set_subgroup(self.symmorphic_indices, tol)

    def set_nonsymmorphic_subgroup(self):

        assert self.symmorphic == False

        if self.nonsymmorphic_indices.shape[0] > 0:
            self.set_subgroup(self.nonsymmorphic_indices)
