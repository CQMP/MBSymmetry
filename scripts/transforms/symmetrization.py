
import numpy as np

from transforms.k_space_transform import filter_matrix


def k_space_symmetrization(mat_vec, stars, star_reps, star_ops, KOrep, k_mesh):

    Nstar = len(star_reps)
    nk = KOrep.shape[0]
    mat_iBZ = []
    mat_all = [[] for i in range(nk)]
    for i in range(Nstar):
        print()
        print(f'star {i}')
        star = stars[i]
        rep = star_reps[i]
        ops = star_ops[i]

        rep_mat = mat_vec[rep]  # matrix for k point in the irreducible wedge
        filter_matrix(rep_mat)
        print('representative:', rep)

        symm_mat = []
        for idx, k in enumerate(star):
            #print(f'point {k}', k_mesh[k].round(10))
            mat = mat_vec[k]
            ops_k = ops[idx]
            #print('symmetry operations:', ops_k)

            for op in ops_k:
                orep = KOrep[rep, op]
                #np.testing.assert_array_almost_equal(orep @ orep.conj().T, np.eye(orep.shape[0]))

                trans_mat = orep.conj().T @ mat @ orep
                #trans_mat[np.abs(trans_mat) < 1e-10] = 0
                symm_mat.append(trans_mat)
        symm_mat = sum(symm_mat) / len(symm_mat)
        norm = np.linalg.norm(symm_mat - rep_mat)
        print('norm of difference at k iBZ:', norm)
        if norm > 1e-3:
            print('Warning: input matrices violate symmetry relation')
        mat_iBZ.append(symm_mat)

        for idx, k in enumerate(star):
            print(f'point {k}', k_mesh[k].round(10))
            ops_k = ops[idx]
            print('symmetry operations:', ops_k)

            mat_k = []
            for op in ops_k:
                orep = KOrep[rep, op]
                #np.testing.assert_array_almost_equal(orep @ orep.conj().T, np.eye(orep.shape[0]))

                trans_mat = orep @ symm_mat @ orep.conj().T
                #trans_mat[np.abs(trans_mat) < 1e-10] = 0
                mat_k.append(trans_mat)
            mat_k = sum(mat_k) / len(mat_k)
            mat_all[k] = mat_k
            norm = np.linalg.norm(mat_vec[k] - mat_k)
            print('norm of difference:', norm)
            if norm > 1e-3:
                print('Warning: input matrices violate symmetry relation')

    return np.stack(mat_all)
