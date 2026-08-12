####################################################################################################
#                                            random.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-12                                                                              #
#                                                                                                  #
# Purpose: Backend-native random numbers: reproducible from a seed, yet fresh on every draw.       #
#          Values are produced by the framework of the reference tensor, so they are created       #
#          directly on its device and never travel through NumPy.                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

import numpy as np

from nifti_mrs_plus.ops import is_jax, is_tf, is_torch

__all__ = ["SeedGenerator"]


#**************************************************************************************************#
#                                       Class SeedGenerator                                        #
#**************************************************************************************************#
#                                                                                                  #
# Backend-native random numbers: reproducible from a seed, yet fresh on every draw.                #
#                                                                                                  #
#**************************************************************************************************#
class SeedGenerator:
    """
    Backend-native random numbers: reproducible from a seed, fresh every draw.

    Augmentation needs two things at once: on-the-fly training wants a
    different perturbation for every batch, and reproducing an experiment wants
    the same whole sequence given a seed.

    The seed fixes the stream and a counter advances once per draw, so
    "normal()" never repeats while the sequence stays a pure function of the
    seed. Hold one generator for the lifetime of the module that draws from it.

    Values are produced by the framework the reference tensor belongs to, so they
    are created directly on its device and never travel through NumPy.

    Note:
        Streams are reproducible per backend, not identical *across* backends:
        NumPy, torch, JAX and TensorFlow have different generators. The same seed
        on the same backend always gives the same sequence.

    Examples:
        >>> rng = SeedGenerator(42)
        >>> a = rng.normal((4,))
        >>> b = rng.normal((4,))          # different draw
        >>> bool((a == b).all())
        False
        >>> bool((SeedGenerator(42).normal((4,)) == a).all())
        True
    """

    def __init__(self, seed=None):
        """
        Args:
            seed: Integer seed for a reproducible stream, or "None" to draw a
                fresh one from OS entropy (still advancing, just not repeatable).
        """
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little")
        self.seed = int(seed) & 0x7FFFFFFFFFFFFFFF
        self._counter = 0

    def _next(self) -> int:
        """Advance the counter and return a seed unique to this draw."""
        self._counter += 1
        # Mix so consecutive counters do not give correlated streams.
        return (self.seed * 6364136223846793005 + self._counter) & 0x7FFFFFFFFFFFFFFF

    def normal(self, shape, like=None, dtype="float32"):
        """Standard normal samples of *shape*, on *like*'s backend and device."""
        key = self._next()

        if like is not None and is_torch(like):
            import torch
            gen = torch.Generator(device=like.device)
            gen.manual_seed(key)
            return torch.randn(tuple(shape), generator=gen, device=like.device,
                               dtype=getattr(torch, dtype))

        if like is not None and is_jax(like):
            import jax
            return jax.random.normal(jax.random.PRNGKey(key & 0xFFFFFFFF),
                                     shape=tuple(shape), dtype=dtype)

        if like is not None and is_tf(like):
            import tensorflow as tf
            return tf.random.stateless_normal(
                tuple(shape), seed=[key & 0x7FFFFFFF, (key >> 31) & 0x7FFFFFFF],
                dtype=dtype)

        return np.random.default_rng(key).standard_normal(tuple(shape)).astype(dtype)

    def uniform(self, shape=(), like=None, low=0.0, high=1.0, dtype="float32"):
        """Uniform samples in "[low, high)", on *like*'s backend and device."""
        key = self._next()

        if like is not None and is_torch(like):
            import torch
            gen = torch.Generator(device=like.device)
            gen.manual_seed(key)
            out = torch.rand(tuple(shape), generator=gen, device=like.device,
                             dtype=getattr(torch, dtype))
            return out * (high - low) + low

        if like is not None and is_jax(like):
            import jax
            return jax.random.uniform(jax.random.PRNGKey(key & 0xFFFFFFFF),
                                      shape=tuple(shape), dtype=dtype,
                                      minval=low, maxval=high)

        if like is not None and is_tf(like):
            import tensorflow as tf
            return tf.random.stateless_uniform(
                tuple(shape), seed=[key & 0x7FFFFFFF, (key >> 31) & 0x7FFFFFFF],
                minval=low, maxval=high, dtype=dtype)

        return np.random.default_rng(key).uniform(low, high, tuple(shape)).astype(dtype)

    def numpy_rng(self) -> np.random.Generator:
        """
        A fresh NumPy generator drawn from this stream.

        For the parameter-space algorithms that have no backend-native form --
        SciPy filter design, B-spline bases, polynomial fits. Those build small
        arrays that are constant with respect to the data, so keeping them in
        NumPy costs no gradient and no device transfer of the data itself; going
        through here keeps them inside the reproducible stream.
        """
        return np.random.default_rng(self._next())

    def __repr__(self):
        return f"SeedGenerator(seed={self.seed}, draws={self._counter})"
