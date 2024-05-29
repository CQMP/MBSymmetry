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


class BLKRSGDFBuilder(_RSGDFBuilder):
    '''
    Use the range-separated algorithm to build Gaussian density fitting 3-center tensor
    '''

    # set True to force calculating j2c^(-1/2) using eigenvalue
    # decomposition (ED); otherwise, Cholesky decomposition (CD) is used
    # first, and ED is called only if CD fails.
    j2c_eig_always = True
    _RSGDFBuilder.j2c_eig_always = True

    def __init__(self, cell, auxcell, kpts=np.zeros((1, 3))):

        _RSGDFBuilder.__init__(self, cell, auxcell, kpts)

    def decompose_j2c(self, j2c):
        j2c = np.asarray(j2c)
        if self.j2c_eig_always:
            return self.eigenvalue_decomposed_metric(j2c)
        else:
            return self.cholesky_decomposed_metric(j2c)

    def cholesky_decomposed_metric(self, j2c):
        try:
            j2c_negative = None
            j2ctag = 'CD'
            j2c = scipy.linalg.cholesky(j2c, lower=True)
            j2c_inv = scipy.linalg.cholesky(j2c, lower=False)  # TODO: this is not really the inverse
        except scipy.linalg.LinAlgError:
            j2c, j2c_negative, j2ctag, j2c_inv = self.eigenvalue_decomposed_metric(j2c)
        return j2c, j2c_negative, j2ctag, j2c_inv

    def eigenvalue_decomposed_metric(self, j2c):
        cell = self.cell
        j2c_negative = None
        w, v = scipy.linalg.eigh(j2c)
        logger.debug(self, 'cond = %.4g, drop %d bfns',
                     w[-1] / w[0], np.count_nonzero(w < self.linear_dep_threshold))
        v1 = v[:, w > self.linear_dep_threshold].conj().T
        v1 /= np.sqrt(w[w > self.linear_dep_threshold]).reshape(-1, 1)
        j2c = v1
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            idx = np.where(w < -self.linear_dep_threshold)[0]
            if len(idx) > 0:
                j2c_negative = (v[:, idx] / np.sqrt(-w[idx])).conj().T
        j2ctag = 'ED'

        v2 = v[:, w > self.linear_dep_threshold]
        v2 *= np.sqrt(w[w > self.linear_dep_threshold]).reshape(1, -1)
        j2c_sqrt = v2

        return j2c, j2c_negative, j2ctag, j2c_sqrt

    def solve_cderi(self, cd_j2c, j3cR, j3cI):
        j2c, j2c_negative, j2ctag, _ = cd_j2c
        if j3cI is None:
            j3c = j3cR.T
        else:
            j3c = (j3cR + j3cI * 1j).T

        cderi_negative = None
        if j2ctag == 'CD':
            cderi = scipy.linalg.solve_triangular(j2c, j3c, lower=True, overwrite_b=True)
        else:
            cderi = lib.dot(j2c, j3c)
            if j2c_negative is not None:
                # for low-dimension systems
                cderi_negative = lib.dot(j2c_negative, j3c)
        return cderi, cderi_negative, j3c

    def gen_uniq_kpts_groups(self, j_only, h5swap, feri=None):
        '''
        Group (kpti,kptj) pairs
        '''
        cpu1 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        if j_only:
            uniq_kpts = np.zeros((1, 3))
            j2c = self.get_2c2e(uniq_kpts)[0]
            cpu1 = log.timer('int2c2e', *cpu1)
            cd_j2c = self.decompose_j2c(j2c)
            j2c = None
            ki = np.arange(nkpts)
            kpt_ii_idx = ki * nkpts + ki
            yield uniq_kpts[0], kpt_ii_idx, cd_j2c

        else:
            uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
                cell, (kpts[None, :, :] - kpts[:, None, :]).reshape(-1, 3))
            scaled_uniq_kpts = cell.get_scaled_kpts(uniq_kpts).round(5)
            log.debug('Num uniq kpts %d', len(uniq_kpts))
            log.debug2('scaled unique kpts %s', scaled_uniq_kpts)

            kpts_idx_pairs = group_by_conj_pairs(cell, uniq_kpts)[0]
            j2c_uniq_kpts = uniq_kpts[[k for k, _ in kpts_idx_pairs]]
            for k, j2c in enumerate(self.get_2c2e(j2c_uniq_kpts)):
                h5swap[f'j2c/{k}'] = j2c
                feri[f'j2c/{k}'] = j2c
                j2c = None
            cpu1 = log.timer('int2c2e', *cpu1)

            # TODO: these are the newly added info
            feri['j2c/uniq_kpts'] = uniq_kpts
            feri['j2c/uniq_index'] = uniq_index
            feri['j2c/uniq_inverse'] = uniq_inverse
            feri['j2c/scaled_uniq_kpts'] = scaled_uniq_kpts
            feri['j2c/j2c_uniq_kpts'] = j2c_uniq_kpts
            feri['j2c/kpts_idx_pairs'] = np.asarray(kpts_idx_pairs, dtype=int)

            for j2c_idx, (k, k_conj) in enumerate(kpts_idx_pairs):
                # Find ki's and kj's that satisfy k_aux = kj - ki
                log.debug1('Cholesky decomposition for j2c at kpt %s %s',
                           k, scaled_uniq_kpts[k])
                j2c = h5swap[f'j2c/{j2c_idx}']
                if k == k_conj:
                    # DF metric for self-conjugated k-point should be real
                    j2c = np.asarray(j2c).real
                cd_j2c = self.decompose_j2c(j2c)

                j2c_sqrt_inv, _, _, j2c_sqrt = cd_j2c
                feri[f'j2c/j2c_sqrt_inv/{j2c_idx}'] = j2c_sqrt_inv
                feri[f'j2c/j2c_sqrt/{j2c_idx}'] = j2c_sqrt

                j2c = None
                kpt_ij_idx = np.where(uniq_inverse == k)[0]
                yield uniq_kpts[k], kpt_ij_idx, cd_j2c

                if k_conj is None or k == k_conj:
                    continue

                # Swap ki, kj for the conjugated case
                log.debug1('Cholesky decomposition for the conjugated kpt %s %s',
                           k_conj, scaled_uniq_kpts[k_conj])
                kpt_ji_idx = np.where(uniq_inverse == k_conj)[0]
                # If self.mesh is not enough to converge compensated charge or
                # SR-coulG, the conj symmetry between j2c[k] and j2c[k_conj]
                # (j2c[k] == conj(j2c[k_conj]) may not be strictly held.
                # Decomposing j2c[k] and j2c[k_conj] may lead to different
                # dimension in cderi tensor. Certain df_ao2mo requires
                # contraction for cderi of k and cderi of k_conj. By using the
                # conj(j2c[k]) and -uniq_kpts[k] (instead of j2c[k_conj] and
                # uniq_kpts[k_conj]), conj-symmetry in j2c is imposed.
                yield -uniq_kpts[k], kpt_ji_idx, _conj_j2c(cd_j2c)

    def make_j3c(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                 j_only=False, shls_slice=None):
        if self.rs_cell is None:
            self.build()
        log = logger.new_logger(self)
        cpu0 = logger.process_clock(), logger.perf_counter()

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao
        naux = self.auxcell.nao
        if shls_slice is None:
            ish0, ish1 = 0, cell.nbas
        else:
            ish0, ish1 = shls_slice[:2]

        dataname = 'j3c'
        fswap = self.outcore_auxe2(cderi_file, intor, aosym, comp, j_only,
                                   dataname, shls_slice)
        cpu1 = log.timer('pass1: real space int3c2e', *cpu0)

        feri = h5py.File(cderi_file, 'w')
        feri['kpts'] = kpts
        feri['aosym'] = aosym
        feri_raw = h5py.File(cderi_file.split('.')[0] + '_raw.h5', 'w')
        feri_raw['kpts'] = kpts
        feri_raw['kmesh'] = cell.get_scaled_kpts(kpts).round(5)
        feri_raw['aosym'] = aosym

        if aosym == 's2':
            nao_pair = nao * (nao + 1) // 2
        else:
            nao_pair = nao ** 2

        if self.has_long_range():
            supmol_ft = _ExtendedMoleFT.from_cell(self.rs_cell, self.bvk_kmesh, verbose=log)
            supmol_ft.exclude_dd_block = self.exclude_dd_block
            supmol_ft = supmol_ft.strip_basis()
            ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False, verbose=log)

        Gv, Gvbase, kws = cell.get_Gv_weights(self.mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]

        def make_cderi(kpt, kpt_ij_idx, j2c):
            log.debug1('make_cderi for %s', kpt)
            log.debug1('kpt_ij_idx = %s', kpt_ij_idx)
            kptjs = kpts[kpt_ij_idx % nkpts]
            nkptj = len(kptjs)
            if self.has_long_range():
                Gaux = self.weighted_ft_ao(kpt)

            mem_now = lib.current_memory()[0]
            log.debug2('memory = %s', mem_now)
            max_memory = max(1000, self.max_memory - mem_now)
            # nkptj for 3c-coulomb arrays plus 1 Lpq array
            buflen = min(max(int(max_memory * .3e6 / 16 / naux / (nkptj + 1)), 1), nao_pair)
            sh_ranges = _guess_shell_ranges(cell, buflen, aosym, start=ish0, stop=ish1)
            buflen = max([x[2] for x in sh_ranges])
            # * 2 for the buffer used in preload
            max_memory -= buflen * naux * (nkptj + 1) * 16e-6 * 2

            # +1 for a pqkbuf
            Gblksize = max(16, int(max_memory * 1e6 / 16 / buflen / (nkptj + 1)))
            Gblksize = min(Gblksize, ngrids, 200000)

            load = self.gen_j3c_loader(fswap, kpt, kpt_ij_idx, aosym)

            cols = [sh_range[2] for sh_range in sh_ranges]
            locs = np.append(0, np.cumsum(cols))
            # buf for ft_aopair
            buf = np.empty(nkptj * buflen * Gblksize, dtype=np.complex128)
            for istep, j3c in enumerate(lib.map_with_prefetch(load, locs[:-1], locs[1:])):
                bstart, bend, ncol = sh_ranges[istep]
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                           istep + 1, len(sh_ranges), bstart, bend, ncol)
                if aosym == 's2':
                    shls_slice = (bstart, bend, 0, bend)
                else:
                    shls_slice = (bstart, bend, 0, cell.nbas)

                if self.has_long_range():
                    for p0, p1 in lib.prange(0, ngrids, Gblksize):
                        # shape of Gpq (nkpts, nGv, ni, nj)
                        Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt,
                                      kptjs, shls_slice, out=buf)
                        self.add_ft_j3c(j3c, Gpq, Gaux, p0, p1)
                        Gpq = None

                j3cR, j3cI = j3c
                for k, kk_idx in enumerate(kpt_ij_idx):
                    cderi, cderi_negative, j3c_ori = self.solve_cderi(j2c, j3cR[k], j3cI[k])
                    feri[f'{dataname}/{kk_idx}/{istep}'] = cderi
                    feri_raw[f'{dataname}/{kk_idx}/{istep}'] = j3c_ori
                    if cderi_negative is not None:
                        # for low-dimension systems
                        feri[f'{dataname}-/{kk_idx}/{istep}'] = cderi_negative
                j3cR = j3cI = j3c = cderi = None

        for kpt, kpt_ij_idx, cd_j2c in self.gen_uniq_kpts_groups(j_only, fswap, feri=feri_raw):
            make_cderi(kpt, kpt_ij_idx, cd_j2c)

        feri.close()
        feri_raw.close()
        cpu1 = log.timer('pass2: AFT int3c2e', *cpu1)
        return self


def _conj_j2c(cd_j2c):
    j2c, j2c_negative, j2ctag, j2c_inv = cd_j2c
    if j2c_negative is None:
        return j2c.conj(), None, j2ctag, j2c_inv.conj()
    else:
        return j2c.conj(), j2c_negative.conj(), j2ctag, j2c_inv.conj()
