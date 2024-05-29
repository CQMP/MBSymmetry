
import numpy as np


def filter_matrix(mat, tol=1e-10):

    mat[np.abs(mat.real) < tol] = 1.j * mat[np.abs(mat.real) < tol].imag
    mat[np.abs(mat.imag) < tol] = mat[np.abs(mat.imag) < tol].real


def transform_matrix(mat_vec, stars, star_reps, star_ops, KOrep, k_mesh, tol=1e-5):

    Nstar = len(star_reps)
    for i in range(Nstar):
        print()
        print(f'star {i}')
        star = stars[i]
        rep = star_reps[i]
        ops = star_ops[i]

        rep_mat = mat_vec[rep]  # matrix for k point in the irreducible wedge
        filter_matrix(rep_mat)
        print('representative:', rep)

        for idx, k in enumerate(star):
            print(f'point {k}', k_mesh[k].round(10))
            mat = mat_vec[k]
            ops_k = ops[idx]
            print('symmetry operations:', ops_k)

            succ = []
            fail = []
            for op in ops_k:
                #print(f'operation {op}')

                orep = KOrep[rep, op]
                np.testing.assert_array_almost_equal(orep @ orep.conj().T, np.eye(orep.shape[0]))

                trans_mat = orep @ rep_mat @ orep.conj().T
                trans_mat[np.abs(trans_mat) < 1e-10] = 0

                if not np.linalg.norm(mat-trans_mat) < tol: #np.allclose(mat, trans_mat, rtol=1.e-5, atol=1.e-8):
                    fail.append(op)
                    filter_matrix(mat)
                    print(np.linalg.norm(mat-trans_mat))
                else:
                    succ.append(op)
            print('succeed operations:', succ)
            print('failed operations:', fail)
            if len(fail) > 0:
                exit(1)


def transform_all(mat_vec, KOrep, trans_table, tol=1e-5):

    print('checking all symmetry operations transform')

    nk, nop, _, _ = KOrep.shape
    for i in range(nk):
        for op in range(nop):
            target = trans_table[i, op]

            mat = mat_vec[i]
            orep = KOrep[i, op]
            trans_mat = orep @ mat @ orep.conj().T
            ref_mat = mat_vec[target]

            if not np.linalg.norm(ref_mat - trans_mat) < tol:
                raise RuntimeError(f'symmetry transformation fails for k point {i}, operation {op}')


def check_little_cogroup_transform(mat_vec, KOrep, little_cogroup, tol=1e-5):

    print('checking little cogroup transform')

    nk, _, _, _ = KOrep.shape
    # check little cogroup of each k point
    for i in range(nk):
        print('point', i)
        ops = little_cogroup[i]
        print('little cogroup', ops)
        for op in ops:
            mat = mat_vec[i]
            orep = KOrep[i, op]
            trans_mat = orep @ mat @ orep.conj().T
            if not np.linalg.norm(mat - trans_mat) < tol:
                raise RuntimeError(f'symmetry transformation of little cogroup fails for k point {i}, operation {op}')
