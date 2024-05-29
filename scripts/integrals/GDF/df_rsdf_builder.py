
import numpy as np
import scipy.linalg
import h5py
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger, zdotCN
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.lib.kpts_helper import (is_zero, member, unique_with_wrap_around, unique,
                                       group_by_conj_pairs)
from pyscf import __config__

from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder, _ExtendedMoleFT
from pyscf.pbc.df.rsdf_builder import LINEAR_DEP_THR
from pyscf.pbc.df.gdf_builder import libpbc, _CCGDFBuilder, _guess_eta
from pyscf.pbc.df.df import GDF

from integrals.GDF.df_rsdf_builder_ref import BLKRSGDFBuilder


class BLKGDF(GDF):

    def __init__(self, cell, kpts=np.zeros((1, 3))):

        GDF.__init__(self, cell, kpts)

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack([self.kpts, self.kpts_band]))[0]

        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            #dfbuilder = _CCGDFBuilder(cell, auxcell, kpts_union)
            #dfbuilder.eta = self.eta
            raise NotImplementedError
        else:
            #dfbuilder = _RSGDFBuilder(cell, auxcell, kpts_union)
            dfbuilder = BLKRSGDFBuilder(cell, auxcell, kpts_union)
        dfbuilder.mesh = self.mesh
        dfbuilder.linear_dep_threshold = self.linear_dep_threshold
        j_only = self._j_only #or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only)
