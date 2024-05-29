
import numpy as np
import scipy.linalg as LA

from pyscf.pbc.df import aft
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.df_jk import zdotCN
from pyscf.pbc.lib.kpts_helper import is_zero, member
from pyscf.pbc.tools import pbc as pbctools
from pyscf import lib


def compute_j2c_sqrt_inv(j2c, j2c_eig_always=False, linear_dep_threshold=1e-9):

    if not j2c_eig_always:
        j2c_sqrt, j2c_sqrt_inv, j2ctag = cholesky_decomposed_metric(j2c)
    else:
        j2c_sqrt, j2c_sqrt_inv, j2ctag = eigenvalue_decomposed_metric(j2c, linear_dep_threshold)

    return j2c_sqrt, j2c_sqrt_inv, j2ctag


def cholesky_decomposed_metric(j2c):
    try:
        j2ctag = 'CD'
        j2c_sqrt = LA.cholesky(j2c, lower=False)
        j2c_sqrt_inv = LA.lapack.clapack.dtrtri(j2c_sqrt)

    except LA.LinAlgError:
        j2c_sqrt, j2c_sqrt_inv, j2ctag = eigenvalue_decomposed_metric(j2c)

    return j2c_sqrt, j2c_sqrt_inv, j2ctag


def eigenvalue_decomposed_metric(j2c, linear_dep_threshold=1e-9):

    w, v = LA.eigh(j2c)
    print("cond = {}, drop {} bfns for lin_dep_threshold = {}".format(w[-1] / w[0],
                                                                      np.count_nonzero(w < linear_dep_threshold),
                                                                      linear_dep_threshold))

    vtrunc = v[:, w > linear_dep_threshold]
    wtrunc = w[w > linear_dep_threshold]
    j2c_sqrt = vtrunc @ np.diag(np.sqrt(wtrunc))  # (naux, ntrunc)
    j2c_sqrt_inv = (vtrunc @ np.diag(1. / np.sqrt(wtrunc))).conj().T
    j2ctag = 'eig'

    #jx = j2c_sqrt @ j2c_sqrt_inv @ j2c
    #print(np.max(np.abs(jx - j2c)))

    return j2c_sqrt, j2c_sqrt_inv, j2ctag


def sqrt_inv_j2c(mydf, j2c, j2c_eig_always=False):
    j2ctags = []
    j2c_sqrt = []
    j2c_sqrt_inv = []
    for iq in range(len(j2c)):
        # j2c_sqrt: (naux_effective, naux). naux_effective <= naux due to linear dependency
        tmp, tmp_inv, tag = compute_j2c_sqrt_inv(j2c[iq], j2c_eig_always, mydf.linear_dep_threshold)
        j2ctags.append(tag)
        j2c_sqrt.append(tmp)
        j2c_sqrt_inv.append(tmp_inv)

    return j2c_sqrt, j2c_sqrt_inv, j2ctags


def _make_j2c_rsgdf(mydf, cell, auxcell, uniq_kpts):

    """
    This function is copied from pyscf rsgdf.py get_2c2e()
    """

    from pyscf.pbc.df import rsdf_helper
    from pyscf.pbc.df.rsdf import get_aux_chg

    # get charge of auxbasis
    if cell.dimension == 3:
        qaux = get_aux_chg(auxcell)
    else:
        qaux = np.zeros(auxcell.nao_nr())

    omega_j2c = abs(mydf.omega_j2c)
    j2c = rsdf_helper.intor_j2c(auxcell, omega_j2c, kpts=uniq_kpts)

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    qaux2 = None
    g0_j2c = np.pi / omega_j2c ** 2. / cell.vol
    mesh_j2c = mydf.mesh_j2c
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh_j2c)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory * .5e6 / 16 / auxcell.nao_nr()))

    for k, kpt in enumerate(uniq_kpts):
        # short-range charge part
        if is_zero(kpt) and cell.dimension == 3:
            if qaux2 is None:
                qaux2 = np.outer(qaux, qaux)
            j2c[k] -= qaux2 * g0_j2c
        # long-range part via aft
        coulG_lr = mydf.weighted_coulG(kpt, False, mesh_j2c, omega_j2c)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
            auxGR = np.asarray(auxG.real, order='C')
            auxGI = np.asarray(auxG.imag, order='C')
            auxG = None

            if is_zero(kpt):  # kpti == kptj
                j2c[k] += lib.ddot(auxGR * coulG_lr[p0:p1], auxGR.T)
                j2c[k] += lib.ddot(auxGI * coulG_lr[p0:p1], auxGI.T)
            else:
                j2cR, j2cI = zdotCN(auxGR * coulG_lr[p0:p1],
                                    auxGI * coulG_lr[p0:p1], auxGR.T, auxGI.T)
                j2c[k] += j2cR + j2cI * 1j
            auxGR = auxGI = j2cR = j2cI = None

        return j2c


def _make_j2c_rsgdf_builder(mydf, cell, auxcell, uniq_kpts):
    from pyscf.pbc.df import gdf_builder

    dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell, uniq_kpts)
    dfbuilder.eta = mydf.eta
    dfbuilder.mesh = mydf.mesh
    dfbuilder.linear_dep_threshold = mydf.linear_dep_threshold
    dfbuilder.build()

    j2c = dfbuilder.get_2c2e(uniq_kpts)

    return j2c


def _make_j2c_gdf_builder(mydf, cell, auxcell, uniq_kpts):
    from pyscf.pbc.df import rsdf_builder

    dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell, uniq_kpts)
    dfbuilder.mesh = mydf.mesh
    dfbuilder.linear_dep_threshold = mydf.linear_dep_threshold
    dfbuilder.build()

    j2c = dfbuilder.get_2c2e(uniq_kpts)

    return j2c
