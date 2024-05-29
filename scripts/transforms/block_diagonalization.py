
import numpy as np
import copy

from transforms.dirac_character import DiracCharacter


def filter_matrix(mat, tol=1e-10):

    mat[np.abs(mat.real) < tol] = 0. + 1.j * mat[np.abs(mat.real) < tol].imag
    mat[np.abs(mat.imag) < tol] = mat[np.abs(mat.imag) < tol].real + 0.j


def check_block_diagonalization(mat, dirac: DiracCharacter, tol):

    size = mat.shape[0]
    n_block = dirac.block_start_idx.shape[0]

    st = dirac.block_start_idx
    en = np.append(dirac.block_start_idx[1:], [size])

    mat_cp = copy.deepcopy(mat)
    for n in range(n_block):
        mat_cp[st[n]:en[n], st[n]:en[n]] *= 0.

    if np.linalg.norm(mat_cp) > tol:
        print(np.linalg.norm(mat_cp))
        return False

    return True


def block_diagonalize_matrix(mat_vec, dirac_characters, check=True, tol=1e-6):

    Nk = len(mat_vec)
    blk_mat_vec = []
    if check: print("check diagonalization")
    for i in range(Nk):
        if check:
            print(f'k point {i}', end=': ')

        dirac = dirac_characters[i]
        filter_matrix(mat_vec[i])
        mat = dirac.U.conj().T @ mat_vec[i] @ dirac.U
        filter_matrix(mat)

        if check:
            succ = check_block_diagonalization(mat, dirac, tol)
            print(succ)

            if succ:
                blk_mat_vec.append(mat)
            else:
                np.savetxt(f'./matrix_{i}_re', mat_vec[i].real, fmt='%.4e')
                np.savetxt(f'./matrix_{i}_im', mat_vec[i].imag, fmt='%.4e')
                raise RuntimeError('block diagonalization fail')
        else:
            blk_mat_vec.append(mat)

    return blk_mat_vec


def flat_block_matrix(blk_mat_vec, dirac_characters):

    # return a single array of flattened mat and index list

    Nk = len(blk_mat_vec)
    flat_mat_vec = []
    for i in range(Nk):
        dirac = dirac_characters[i]
        flat_mat = []
        blk_mat = blk_mat_vec[i]
        size = blk_mat.shape[0]
        n_block = dirac.block_start_idx.shape[0]
        st = dirac.block_start_idx
        en = np.append(dirac.block_start_idx[1:], [size])
        for n in range(n_block):
            blk = blk_mat[st[n]:en[n], st[n]:en[n]].reshape(-1)
            flat_mat.append(blk)

        flat_mat = np.concatenate(flat_mat)
        flat_mat_vec.append(flat_mat)

    flat_mat_vec = np.concatenate(flat_mat_vec)

    return flat_mat_vec
