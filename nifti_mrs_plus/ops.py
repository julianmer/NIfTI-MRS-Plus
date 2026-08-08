####################################################################################################
#                                              ops.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-27                                                                              #
#                                                                                                  #
# Purpose: Backend-agnostic tensor operations for MRS signal processing. Dispatches on the type of #
#          the tensor you already hold, so callers write one expression that runs on numpy,        #
#          torch, jax or tensorflow. Follows the zea philosophy: write once, run anywhere.         #
#                                                                                                  #
#          The name "Backend" in core.py means a storage format; this module is the op namespace.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

import numpy as np


#***********************#
#   backend detection   #
#***********************#
def is_torch(x):
    """True if *x* is a PyTorch tensor."""
    return type(x).__module__.split(".")[0] == "torch"


def is_jax(x):
    """True if *x* is a JAX array."""
    return type(x).__module__.split(".")[0] in ("jax", "jaxlib")


def is_tf(x):
    """True if *x* is a TensorFlow tensor."""
    return type(x).__module__.split(".")[0] == "tensorflow"


def namespace(x):
    """
    The array module *x* belongs to: "torch", "jax.numpy", "tensorflow"
    or "numpy".

    Dispatch is on the tensor rather than on a global setting, so several
    backends can be in use at once in one process and data is never moved
    between frameworks behind the caller's back.

    Keras needs no branch of its own: "keras.ops" returns the active backend's
    native type, so Keras data lands on the right branch here automatically.
    """
    if is_torch(x):
        import torch
        return torch
    if is_jax(x):
        import jax.numpy as jnp
        return jnp
    if is_tf(x):
        import tensorflow as tf
        return tf
    return np


#******************#
#   construction   #
#******************#
def arange_like(ref, n, dtype="float32"):
    """"arange(n)" on the same backend, and for torch the same device, as *ref*.

    Creating an array needs a reference tensor rather than a global setting,
    because the result has to meet the data it will be combined with. A *ref* of
    None means NumPy, for coordinates built before any tensor exists.
    """
    if is_torch(ref):
        import torch
        return torch.arange(n, dtype=getattr(torch, dtype), device=ref.device)
    if is_tf(ref):
        import tensorflow as tf
        return tf.range(n, dtype=dtype)
    return namespace(ref).arange(n, dtype=dtype)


def linspace_like(ref, start, stop, num, endpoint=True, dtype="float32"):
    """Evenly spaced values on *ref*'s backend, like "numpy.linspace"."""
    num = int(num)
    if is_torch(ref):
        import torch
        tdtype = getattr(torch, dtype)
        if endpoint:
            return torch.linspace(start, stop, num, dtype=tdtype, device=ref.device)
        # torch always includes the stop value, so make one extra and drop it
        return torch.linspace(start, stop, num + 1, dtype=tdtype, device=ref.device)[:-1]
    if is_tf(ref):
        import tensorflow as tf
        if endpoint:
            return tf.cast(tf.linspace(start, stop, num), dtype)
        return tf.cast(tf.linspace(start, stop, num + 1)[:-1], dtype)
    return namespace(ref).linspace(start, stop, num, endpoint=endpoint, dtype=dtype)


def full_like_shape(ref, shape, fill_value, dtype="float32"):
    """An array of *shape* filled with *fill_value*, on *ref*'s backend."""
    if is_torch(ref):
        import torch
        return torch.full(tuple(shape), fill_value,
                          dtype=getattr(torch, dtype), device=ref.device)
    if is_tf(ref):
        import tensorflow as tf
        return tf.fill(tuple(shape), tf.cast(fill_value, dtype))
    return namespace(ref).full(tuple(shape), fill_value, dtype=dtype)


def asarray_like(ref, data, dtype="float32"):
    """Convert *data* to an array on *ref*'s backend."""
    if is_torch(ref):
        import torch
        if isinstance(data, torch.Tensor):
            return data.to(dtype=getattr(torch, dtype), device=ref.device)
        return torch.as_tensor(np.asarray(data), dtype=getattr(torch, dtype),
                               device=ref.device)
    if is_tf(ref):
        import tensorflow as tf
        return tf.cast(tf.convert_to_tensor(data), dtype)
    if is_jax(ref):
        import jax.numpy as jnp
        return jnp.asarray(data, dtype=dtype)
    return np.asarray(data, dtype=dtype)


#****************#
#   arithmetic   #
#****************#
def exp(x):
    """Element-wise exponential."""
    return namespace(x).exp(x)


def sqrt(x):
    """Element-wise square root."""
    return namespace(x).sqrt(x)


def sin(x):
    """Element-wise sine."""
    return namespace(x).sin(x)


def cos(x):
    """Element-wise cosine."""
    return namespace(x).cos(x)


def floor(x):
    """Element-wise floor."""
    return namespace(x).floor(x)


def ceil(x):
    """Element-wise ceiling."""
    return namespace(x).ceil(x)


def abs(x):                                          # noqa: A001 - mirrors numpy
    """Element-wise magnitude. Complex in, real out."""
    return namespace(x).abs(x)


def where(condition, a, b):
    """Element-wise select, like "numpy.where"."""
    return namespace(a).where(condition, a, b)


def real(x):
    """Real part."""
    if is_tf(x):
        import tensorflow as tf
        return tf.math.real(x)
    return namespace(x).real(x)


def imag(x):
    """Imaginary part."""
    if is_tf(x):
        import tensorflow as tf
        return tf.math.imag(x)
    return namespace(x).imag(x)


def is_complex(x):
    """True if *x* has a complex dtype, on any backend."""
    if is_torch(x):
        import torch
        return torch.is_complex(x)
    if is_tf(x):
        import tensorflow as tf
        return x.dtype.is_complex
    return bool(np.iscomplexobj(x))


def transpose(x, perm):
    """Permute the axes of *x*."""
    if is_torch(x):
        return x.permute(*perm)
    if is_tf(x):
        import tensorflow as tf
        return tf.transpose(x, perm)
    return namespace(x).transpose(x, perm)


def sum(x, axis=None, keepdims=False):               # noqa: A001 - mirrors numpy
    """Sum over *axis* (all axes when "None")."""
    if is_torch(x):
        import torch
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis, keepdim=keepdims)
    if is_tf(x):
        import tensorflow as tf
        return tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    return namespace(x).sum(x, axis=axis, keepdims=keepdims)


def pad(x, width):
    """Pad *x*; *width* is a per-axis "(before, after)" sequence."""
    if is_torch(x):
        import torch
        flat = []
        for before, after in reversed(list(width)):
            flat.extend([int(before), int(after)])
        return torch.nn.functional.pad(x, flat)
    if is_tf(x):
        import tensorflow as tf
        return tf.pad(x, [list(w) for w in width])
    return namespace(x).pad(x, [tuple(w) for w in width])


def shift_right(x, n):
    """
    Shift *x* along its last axis by *n* samples, filling with zeros.

    A delayed copy of a FID, "out[..., n:] = x[..., :-n]", expressed as a pad
    and a slice so it stays on the tensor's own backend and keeps its gradient
    rather than going through NumPy assignment.
    """
    if n <= 0:
        return x
    length = int(x.shape[-1])
    if n >= length:
        return x * 0

    if is_torch(x):
        import torch
        # torch pads the last axis first, as (left, right)
        return torch.nn.functional.pad(x, (n, 0))[..., :length]

    pad_width = [(0, 0)] * (len(x.shape) - 1) + [(n, 0)]
    if is_tf(x):
        import tensorflow as tf
        return tf.pad(x, pad_width)[..., :length]
    return namespace(x).pad(x, pad_width)[..., :length]


def complex_from(re, im):
    """Build a complex tensor from real and imaginary parts."""
    if is_torch(re):
        import torch
        return torch.complex(re, im)
    if is_tf(re):
        import tensorflow as tf
        return tf.complex(re, im)
    return re + 1j * im


#***************#
#   reductions  #
#***************#
# torch spells the axis `dim`/`keepdim` and rejects `dim=None`; TensorFlow puts
# reductions under `reduce_*`. numpy and jax already agree, so they fall through.

def amax(x, axis=None, keepdims=False):
    """Maximum over *axis* (all axes when "None")."""
    if is_torch(x):
        import torch
        if axis is None:
            return torch.amax(x)
        return torch.amax(x, dim=axis, keepdim=keepdims)
    if is_tf(x):
        import tensorflow as tf
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)
    return namespace(x).max(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    """Mean over *axis* (all axes when "None")."""
    if is_torch(x):
        import torch
        if axis is None:
            return torch.mean(x)
        return torch.mean(x, dim=axis, keepdim=keepdims)
    if is_tf(x):
        import tensorflow as tf
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    return namespace(x).mean(x, axis=axis, keepdims=keepdims)


#***************#
#   reshaping   #
#***************#
def clip(x, lo, hi):
    """Clamp *x* into [lo, hi]."""
    if is_torch(x):
        import torch
        return torch.clamp(x, lo, hi)
    if is_tf(x):
        import tensorflow as tf
        return tf.clip_by_value(x, lo, hi)
    return namespace(x).clip(x, lo, hi)


def take(x, indices, axis):
    """Select *indices* along *axis*, like numpy.take."""
    if is_torch(x):
        import torch
        idx = indices if is_torch(indices) else torch.as_tensor(
            np.asarray(indices), device=x.device)
        return torch.index_select(x, axis, idx.long())
    if is_tf(x):
        import tensorflow as tf
        return tf.gather(x, indices, axis=axis)
    return namespace(x).take(x, indices, axis=axis)


def take_along(x, indices, axis):
    """Gather along *axis* with per-element *indices*, like numpy.take_along_axis."""
    if is_torch(x):
        import torch
        return torch.gather(x, axis, indices)
    if is_tf(x):
        import tensorflow as tf
        return tf.gather(x, indices, axis=axis, batch_dims=axis)
    return namespace(x).take_along_axis(x, indices, axis=axis)


def shape(x):
    """Shape of *x* as a plain tuple of ints, on any backend."""
    return tuple(int(d) for d in x.shape)


def reshape(x, shape):
    """Reshape *x*."""
    return namespace(x).reshape(x, shape)


def stack(tensors, axis=0):
    """Stack *tensors* along a new *axis*."""
    first = tensors[0]
    if is_torch(first):
        import torch
        return torch.stack(tensors, dim=axis)
    return namespace(first).stack(tensors, axis=axis)


def concatenate(tensors, axis=-1):
    """
    Join *tensors* along *axis*.

    Chunked processing has to collect its pieces and join them: jax and
    tensorflow tensors are immutable, so writing slices into a preallocated
    output is not available on every backend.
    """
    first = tensors[0]
    if is_torch(first):
        import torch
        return torch.cat(tensors, dim=axis)
    if is_tf(first):
        import tensorflow as tf
        return tf.concat(tensors, axis=axis)
    return namespace(first).concatenate(tensors, axis=axis)


def cast(x, dtype):
    """Cast *x* to *dtype*, given as a string such as "float32" or "bool"."""
    if is_torch(x):
        import torch
        return x.to(getattr(torch, dtype))
    if is_tf(x):
        import tensorflow as tf
        return tf.cast(x, dtype)
    return x.astype(dtype)


def cast_bool(x):
    """Cast *x* to boolean."""
    return cast(x, "bool")


def cast_like(x, ref):
    """Cast *x* to *ref*'s dtype.

    TensorFlow refuses to multiply a complex64 tensor by a float32 one, so a
    real envelope must be promoted before it meets complex data. The other
    backends promote silently; doing it explicitly everywhere keeps one code
    path instead of a TensorFlow special case at every call site.
    """
    if is_torch(x):
        return x.to(ref.dtype)
    if is_tf(x):
        import tensorflow as tf
        return tf.cast(x, ref.dtype)
    return x.astype(ref.dtype)


#************************#
#   complex fft / ifft   #
#************************#
# Every backend supports native complex-valued FFT with the same semantics, so
# we dispatch to the right module. This is the one family keras.ops cannot
# cover: keras.ops.fft takes a (real, imag) tuple and rejects complex tensors,
# which would force every caller to split and rejoin around each transform.

def fftn(x, axes, norm="ortho"):
    """
    Complex FFT over several axes at once, for spatial k-space transforms.

    TensorFlow only transforms its own last axes and has no norm argument, so
    there the requested axes are moved to the end, transformed, scaled and moved
    back. The other backends take axes directly.

    Args:
        x: Complex tensor.
        axes: Axes to transform, as a tuple.
        norm: FFT normalization, as in numpy.

    Returns:
        Complex tensor with the same shape as *x*.
    """
    return _fftn(x, tuple(axes), norm, inverse=False)


def ifftn(x, axes, norm="ortho"):
    """Inverse of :func:`fftn`."""
    return _fftn(x, tuple(axes), norm, inverse=True)


def _fftn(x, axes, norm, inverse):
    """Shared body of fftn / ifftn."""
    if is_torch(x):
        import torch
        fn = torch.fft.ifftn if inverse else torch.fft.fftn
        return fn(x, dim=axes, norm=norm)
    if is_jax(x):
        import jax.numpy as jnp
        fn = jnp.fft.ifftn if inverse else jnp.fft.fftn
        return fn(x, axes=axes, norm=norm)
    if is_tf(x):
        return _tf_fftn(x, axes, norm, inverse)

    fn = np.fft.ifftn if inverse else np.fft.fftn
    return fn(x, axes=axes, norm=norm)


def _tf_fftn(x, axes, norm, inverse):
    """fftn for TensorFlow, which transforms only its trailing axes."""
    import tensorflow as tf

    if len(axes) not in (1, 2, 3):
        raise ValueError(
            f"TensorFlow can transform 1, 2 or 3 axes at a time, got {len(axes)}."
        )

    rank = len(x.shape)
    axes = tuple(a % rank for a in axes)
    trailing = tuple(range(rank - len(axes), rank))

    moved = axes != trailing
    if moved:
        rest = [a for a in range(rank) if a not in axes]
        forward_perm = rest + list(axes)
        inverse_perm = [forward_perm.index(a) for a in range(rank)]
        x = tf.transpose(x, forward_perm)

    fn = {1: (tf.signal.ifft, tf.signal.fft),
          2: (tf.signal.ifft2d, tf.signal.fft2d),
          3: (tf.signal.ifft3d, tf.signal.fft3d)}[len(axes)][not inverse]
    out = fn(x)

    # tf.signal normalizes like norm=None: no scaling forward, 1/n inverse.
    # After the transpose the transformed axes are the trailing ones.
    n = 1
    for a in trailing:
        n *= int(x.shape[a])

    if norm == "ortho":
        scale = tf.cast(np.sqrt(n), out.dtype)
        out = out * scale if inverse else out / scale
    elif norm == "forward":
        factor = tf.cast(n, out.dtype)
        out = out * factor if inverse else out / factor

    if moved:
        out = tf.transpose(out, inverse_perm)
    return out


def fft(x):
    """Complex-to-complex FFT along the last axis (any backend).

    MRS convention: "fft" maps *spectrum -> FID*.

    Args:
        x: Complex tensor of any shape.

    Returns:
        Complex tensor with the same shape as *x*.
    """
    if is_torch(x):
        import torch
        return torch.fft.fft(x)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.fft(x)
    if is_tf(x):
        import tensorflow as tf
        return tf.signal.fft(x)
    return np.fft.fft(x)


def ifft(x):
    """Complex-to-complex inverse FFT along the last axis (any backend).

    MRS convention: "ifft" maps *FID -> spectrum*.

    Args:
        x: Complex tensor of any shape.

    Returns:
        Complex tensor with the same shape as *x*.
    """
    if is_torch(x):
        import torch
        return torch.fft.ifft(x)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.ifft(x)
    if is_tf(x):
        import tensorflow as tf
        return tf.signal.ifft(x)
    return np.fft.ifft(x)


#**************************#
#   fftshift / ifftshift   #
#**************************#
# numpy / jax use "axes=" keyword; torch uses "dim=". TensorFlow has no
# fftshift at all, so it is expressed as a roll by half the axis length --
# which is exactly what fftshift is.

def fftshift(x, axis=-1):
    """Shift zero-frequency component to the center (like "np.fft.fftshift").

    Args:
        x: Tensor of any shape.
        axis: Axis (or axes) along which to shift.

    Returns:
        Shifted tensor.
    """
    if is_torch(x):
        import torch
        return torch.fft.fftshift(x, dim=axis)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.fftshift(x, axes=axis)
    if is_tf(x):
        return _tf_roll(x, axis, inverse=False)
    return np.fft.fftshift(x, axes=axis)


def ifftshift(x, axis=-1):
    """Inverse of "fftshift" (like "np.fft.ifftshift").

    Args:
        x: Tensor of any shape.
        axis: Axis (or axes) along which to shift.

    Returns:
        Shifted tensor.
    """
    if is_torch(x):
        import torch
        return torch.fft.ifftshift(x, dim=axis)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.ifftshift(x, axes=axis)
    if is_tf(x):
        return _tf_roll(x, axis, inverse=True)
    return np.fft.ifftshift(x, axes=axis)


def _tf_roll(x, axis, inverse):
    """fftshift / ifftshift for TensorFlow, which ships neither.

    "fftshift" rolls by "n // 2"; "ifftshift" rolls back by "n - n // 2".
    The two differ only for odd "n", which is why they cannot share a shift.
    """
    import tensorflow as tf

    axes = [axis] if isinstance(axis, int) else list(axis)
    shape = tf.shape(x)
    rank = len(x.shape)
    shifts = []
    for ax in axes:
        n = shape[ax % rank]
        shifts.append(n - n // 2 if inverse else n // 2)
    return tf.roll(x, shift=tf.stack(shifts), axis=[ax % rank for ax in axes])


#************************#
#   conversion helpers   #
#************************#
def to_numpy(x):
    """Convert *any* backend tensor to a NumPy "ndarray".

    Handles PyTorch ".detach().cpu()" and JAX/TF device arrays
    transparently.

    Note:
        This detaches. It is the boundary at which an autograd graph ends, so
        call it only where a numpy array is genuinely required -- writing to
        NIfTI, plotting, assertions -- never in the middle of a pipeline.

    Args:
        x: Tensor or array.

    Returns:
        "numpy.ndarray"
    """
    if isinstance(x, np.ndarray):
        return x
    if is_torch(x):
        return x.detach().cpu().numpy()
    # JAX, TF, and anything with __array__ protocol
    return np.asarray(x)


def match_backend(param, ref):
    """Convert a NumPy parameter array to the same backend as *ref*.

    Promotes a lightweight NumPy parameter -- an envelope, a phase ramp, a
    baseline -- onto whatever tensor the caller holds::

        result = data_tensor * match_backend(envelope_np, data_tensor)

    This avoids the known issue where "torch.Tensor * np.ndarray" can
    fail when "requires_grad=True" (numpy tries to call "__array__"
    on the grad-enabled tensor via ufunc dispatch).

    For NumPy-to-NumPy this is a no-op (zero cost).

    Note:
        Use this for values that are *constant with respect to the data*, where
        it costs nothing. It promotes the result of a computation, not the
        computation itself, so a kernel built entirely this way never runs on
        the accelerator. Prefer "keras.ops" for the kernel itself.

    Args:
        param: Parameter array (typically numpy).
        ref:   Reference tensor whose backend we want to match.

    Returns:
        "param" converted to the same framework and dtype as "ref".
    """
    if isinstance(param, np.ndarray):
        # Always adopt ref's dtype. A float64 NumPy envelope meeting complex64
        # data would otherwise promote the result to complex128, doubling memory
        # for the rest of the pipeline; TensorFlow refuses the mixture outright.
        if isinstance(ref, np.ndarray):
            return param.astype(ref.dtype)
        if is_torch(ref):
            import torch
            return torch.as_tensor(param, device=ref.device).to(ref.dtype)
        if is_jax(ref):
            import jax.numpy as jnp
            return jnp.array(param, dtype=ref.dtype)
        if is_tf(ref):
            import tensorflow as tf
            return tf.constant(param, dtype=ref.dtype)
    # Already same backend, or unknown -> return as-is
    return param


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


__all__ = [
    "fft", "ifft", "fftshift", "ifftshift",
    "to_numpy", "match_backend",
    "is_torch", "is_jax", "is_tf", "namespace",
    "arange_like", "linspace_like", "full_like_shape", "asarray_like",
    "exp", "sqrt", "sin", "cos", "floor", "ceil", "abs", "where", "real",
    "complex_from", "imag", "is_complex", "transpose",
    "stack", "cast", "cast_bool", "shape", "clip", "take", "take_along", "scatter_add", "affine_grid", "grid_sample",
    "fftn", "ifftn", "concatenate",
    "shift_right",
    "amax", "mean", "sum", "pad", "reshape", "cast_like",
    "SeedGenerator",
]


#*******************#
#   resampling      #
#*******************#
# affine_grid and grid_sample mirror their torch.nn.functional namesakes with
# align_corners=False and bilinear/trilinear interpolation, expressed as a
# gather so every backend can run them and gradients flow to the sampled data.

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
