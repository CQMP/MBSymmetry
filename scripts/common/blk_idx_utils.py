
import numpy as np
import h5py


def find_blk_index(k_diracs):

    # kao_slice_offsets, kao_slice_sizes -> (list of nk)
    # kao_nblocks -> (number of blocks at each momentum)
    # ao_offsets, ao_sizes -> (list of list)
    # ao_block_offsets

    nk = len(k_diracs)
    k_offsets = 0
    ao_offsets, ao_sizes, ao_block_offsets = [], [], []
    kao_nblocks, kao_slice_offsets, kao_slice_sizes = [], [], []
    for k in range(nk):
        ao_sizes.append(k_diracs[k].block_size)
        ao_block_offsets.append(k_diracs[k].block_start_idx)
        nblock = len(k_diracs[k].block_size)
        kao_nblocks.append(nblock)
        offsets = [0]
        for i in range(1, nblock):
            offsets.append(offsets[i-1]+k_diracs[k].block_size[i-1]**2)
        ao_offsets.append(offsets)

        kao_slice_offsets.append(k_offsets)
        slice_size = offsets[-1]+k_diracs[k].block_size[-1]**2
        k_offsets += slice_size
        kao_slice_sizes.append(slice_size)

    return kao_slice_offsets, kao_slice_sizes, kao_nblocks, ao_offsets, ao_sizes, ao_block_offsets


def save_blk_index(filename, kao_slice_offsets, kao_slice_sizes, kao_nblocks,
                   ao_offsets, ao_sizes, ao_block_offsets, prefix='ao'):

    kprefix = 'kao' if prefix == 'ao' else 'qaux'
    f = h5py.File(filename, 'a')
    g = f.require_group('block')
    g[f'{kprefix}_slice_offsets'] = kao_slice_offsets
    g[f'{kprefix}_slice_sizes'] = kao_slice_sizes
    g[f'{kprefix}_nblocks'] = kao_nblocks
    nk = len(kao_nblocks)
    for i in range(nk):
        g[f'{prefix}_offsets/{i}'] = ao_offsets[i]
        g[f'{prefix}_sizes/{i}'] = ao_sizes[i]
        g[f'{prefix}_block_offsets/{i}'] = ao_block_offsets[i]
    f.close()


def save_kpair_blk_index(filename, kpair_slice_offsets, kpair_slice_sizes, kpair_nblk, tensor_offsets, tensor_irreps):

    f = h5py.File(filename, 'a')
    g = f.require_group('block')
    g['kpair_slice_offsets'] = kpair_slice_offsets
    g['kpair_slice_sizes'] = kpair_slice_sizes
    g['kpair_nblocks'] = kpair_nblk
    nkpair = len(kpair_nblk)
    for i in range(nkpair):
        g[f'tensor_offsets/{i}'] = tensor_offsets[i]
        g[f'tensor_irreps/{i}'] = tensor_irreps[i]
    f.close()


def save_orep_blk_info(filename, orep_blk, orep_flat, orep_irreps, orep_offsets, orep_sizes, name='orep'):

    blk_name = 'KspaceOrepBlk' if name == 'orep' else 'KspaceAuxRepTransBlk'
    group_name = 'kao_rot' if name == 'orep' else 'kaux_rot'

    f = h5py.File(filename, 'a')
    f[blk_name] = orep_blk.view(np.float64).reshape(*orep_blk.shape, 2)
    f[blk_name].attrs["__complex__"] = np.int8(1)
    g = f.require_group(group_name)
    nk = len(orep_flat)
    for i in range(nk):
        g[f'{name}_flat/{i}'] = orep_flat[i].view(np.float64).reshape(*orep_flat[i].shape, 2)
        g[f'{name}_flat/{i}'].attrs["__complex__"] = np.int8(1)
        g[f'{name}_irreps/{i}'] = orep_irreps[i]
        g[f'{name}_offsets/{i}'] = orep_offsets[i]
        g[f'{name}_sizes/{i}'] = orep_sizes[i]
    f.close()
