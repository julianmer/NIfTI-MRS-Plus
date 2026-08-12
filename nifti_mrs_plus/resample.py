####################################################################################################
#                                           resample.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-12                                                                              #
#                                                                                                  #
# Purpose: Backend-agnostic resampling, composed from the dispatch primitives in ops.py.           #
#          affine_grid and grid_sample mirror their torch.nn.functional namesakes with             #
#          align_corners=False and bilinear/trilinear interpolation, expressed as a gather so      #
#          every backend can run them and gradients flow to the sampled data. scatter_add is       #
#          the gridding half of a NUFFT.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from nifti_mrs_plus.ops import (arange_like, cast, cast_like, clip, floor, full_like_shape,
                                is_jax, is_tf, is_torch, reshape, shape, stack, take_along)

__all__ = ["affine_grid", "grid_sample", "scatter_add"]


#*******************#
#   resampling      #
#*******************#
def affine_grid(theta, size):
    """
    Sampling grid for an affine transform, like "torch.nn.functional.affine_grid".

    Args:
        theta: Affine matrices, "(N, 2, 3)" for 2-D or "(N, 3, 4)" for 3-D.
        size: Target shape, "(N, C, H, W)" or "(N, C, D, H, W)".

    Returns:
        Grid of normalized coordinates in [-1, 1], "(N, H, W, 2)" or
        "(N, D, H, W, 3)", ordered x, y[, z] to match torch.
    """
    spatial = tuple(int(s) for s in size[2:])
    n_dim = len(spatial)

    # Normalized centers, matching align_corners=False
    axes = [(2.0 * arange_like(theta, n) + 1.0) / n - 1.0 for n in spatial]

    # torch orders the grid's last axis x, y, z, i.e. the reverse of the
    # spatial axes, so build the mesh reversed and stack back in that order.
    grids = []
    for i, axis in enumerate(axes):
        view = [1] * n_dim
        view[i] = spatial[i]
        broadcast = reshape(axis, view)
        for j, n in enumerate(spatial):
            if j != i:
                broadcast = broadcast + full_like_shape(theta, [1] * n_dim, 0.0)
        grids.append(broadcast + full_like_shape(theta, spatial, 0.0))

    coords = stack(grids[::-1] + [full_like_shape(theta, spatial, 1.0)], axis=-1)
    coords = reshape(coords, (-1, n_dim + 1))                    # (prod(spatial), n_dim+1)

    # (N, n_dim, n_dim+1) x (n_dim+1, P) -> (N, P, n_dim)
    out = []
    for n in range(int(theta.shape[0])):
        rows = [
            reshape(
                sum(coords[:, k] * theta[n, d, k] for k in range(n_dim + 1)),
                (-1,),
            )
            for d in range(n_dim)
        ]
        out.append(stack(rows, axis=-1))

    return reshape(stack(out, axis=0), (int(theta.shape[0]),) + spatial + (n_dim,))


def grid_sample(x, grid, padding_mode="zeros"):
    """
    Resample *x* at *grid*, like "torch.nn.functional.grid_sample".

    Bilinear for 2-D, trilinear for 3-D, with align_corners=False. Implemented
    as a gather with corner weights, so it runs on every backend and gradients
    reach the sampled data.

    Args:
        x: "(N, C, H, W)" or "(N, C, D, H, W)", real valued.
        grid: Normalized coordinates from :func:"affine_grid".
        padding_mode: "zeros" or "border".

    Returns:
        Resampled tensor shaped "(N, C) + grid.shape[1:-1]".
    """
    spatial = shape(x)[2:]
    n_dim = len(spatial)
    n_batch, n_chan = shape(x)[0], shape(x)[1]
    out_spatial = shape(grid)[1:-1]
    n_points = 1
    for s in out_spatial:
        n_points *= s

    flat_grid = reshape(grid, (n_batch, n_points, n_dim))

    # grid orders coordinates x, y, z; spatial axes run z, y, x
    unnorm = []
    for d in range(n_dim):
        length = spatial[n_dim - 1 - d]
        unnorm.append(((flat_grid[..., d] + 1.0) * length - 1.0) / 2.0)

    # accumulate the 2**n_dim corners
    result = None
    for corner in range(2 ** n_dim):
        weight, flat_index = None, None
        for d in range(n_dim):
            length = spatial[n_dim - 1 - d]
            low = floor(unnorm[d])
            frac = unnorm[d] - low
            take_high = (corner >> d) & 1
            idx = low + 1.0 if take_high else low
            w = frac if take_high else 1.0 - frac

            if padding_mode == "zeros":
                # torch drops each out-of-range corner on its own, without
                # renormalising the remaining weights
                w = w * cast_like((idx >= 0) & (idx <= length - 1), w)

            idx = clip(idx, 0, length - 1)
            weight = w if weight is None else weight * w
            # spatial axis for grid dim d is (n_dim - 1 - d)
            stride = 1
            for k in range(n_dim - d, n_dim):
                stride *= spatial[k]
            flat_index = idx * stride if flat_index is None else flat_index + idx * stride

        index = cast(flat_index, "int64")
        index = reshape(index, (n_batch, 1, n_points))
        index = index + full_like_shape(index, (n_batch, n_chan, n_points), 0, dtype="int64")

        gathered = take_along(reshape(x, (n_batch, n_chan, -1)), index, axis=2)
        contribution = gathered * cast_like(reshape(weight, (n_batch, 1, n_points)), gathered)
        result = contribution if result is None else result + contribution

    return reshape(result, (n_batch, n_chan) + tuple(out_spatial))


def scatter_add(shape, indices, values):
    """
    Accumulate *values* at flat *indices* into a zero array of *shape*.

    The gridding half of a NUFFT: many samples land in the same Cartesian bin,
    so contributions must add rather than overwrite. Every backend spells this
    differently, and only this form accumulates duplicates correctly.

    Args:
        shape: Shape of the output.
        indices: Flat integer indices into the first axis of the output.
        values: Values to accumulate, one per index.

    Returns:
        Array of *shape* on *values*' backend.
    """
    if is_torch(values):
        import torch
        out = torch.zeros(tuple(shape), dtype=values.dtype, device=values.device)
        idx = indices if is_torch(indices) else torch.as_tensor(np.asarray(indices),
                                                               device=values.device)
        return out.index_add(0, idx.long(), values)

    if is_jax(values):
        import jax.numpy as jnp
        out = jnp.zeros(tuple(shape), dtype=values.dtype)
        return out.at[jnp.asarray(indices)].add(values)

    if is_tf(values):
        import tensorflow as tf
        return tf.tensor_scatter_nd_add(
            tf.zeros(tuple(shape), dtype=values.dtype),
            tf.reshape(tf.cast(indices, tf.int32), (-1, 1)),
            values)

    out = np.zeros(tuple(shape), dtype=values.dtype)
    np.add.at(out, np.asarray(indices), values)
    return out
