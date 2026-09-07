####################################################################################################
#                                            backend.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-03                                                                              #
#                                                                                                  #
# Purpose: Backend-agnostic function transforms: jit compilation and value-and-gradient            #
#          evaluation. Where ops.py dispatches element operations on the tensor you hold, this     #
#          module dispatches whole functions at call time on the tensors you pass, so one wrapped  #
#          function serves numpy, torch, jax, and tensorflow arguments in the same process.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from nifti_mrs_plus.ops import is_jax, is_tf, is_torch


#*******************#
#   jit dispatch    #
#*******************#
def jit(fn):
    """Wrap `fn` so it is compiled for whichever framework its tensor arguments belong to.

    The framework is detected at call time from the first recognized tensor argument and the
    compiled variant is cached per framework, so the same wrapper serves torch, jax, and
    tensorflow tensors side by side. Calls without a recognized tensor (e.g. numpy) run `fn`
    uncompiled.

    Args:
        fn: A pure, tensor-in/tensor-out function.

    Returns:
        The dispatching wrapper.
    """
    compiled = {}

    def wrapped(*args, **kwargs):
        name = _framework_of(args)
        if name not in compiled:
            compiled[name] = _compile(fn, name)
        return compiled[name](*args, **kwargs)

    return wrapped


#**********************#
#   value and grad     #
#**********************#
def value_and_grad(fn):
    """Wrap a real-scalar function so calls return `(value, gradient)` for its first argument.

    The framework is detected at call time from the first argument, which must be a real
    tensor of the framework you want the gradient in. Remaining arguments are passed through
    unchanged.

    Args:
        fn: A function returning a real scalar tensor, differentiable in its first argument.

    Returns:
        The dispatching wrapper.

    Raises:
        NotImplementedError: When called with a numpy (or unrecognized) first argument.
    """
    cache = {}

    def wrapped(x, *args, **kwargs):
        if is_jax(x):
            if "jax" not in cache:
                import jax

                cache["jax"] = jax.value_and_grad(fn)
            return cache["jax"](x, *args, **kwargs)
        if is_torch(x):
            import torch

            leaf = x.detach().clone().requires_grad_(True)
            value = fn(leaf, *args, **kwargs)
            (grad,) = torch.autograd.grad(value, (leaf,))
            return value.detach(), grad
        if is_tf(x):
            import tensorflow as tf

            leaf = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(leaf)
                value = fn(leaf, *args, **kwargs)
            return value, tape.gradient(value, leaf)
        raise NotImplementedError(
            "value_and_grad needs a torch, jax, or tensorflow tensor as the first argument"
        )

    return wrapped


#**********************#
#   framework lookup   #
#**********************#
def _framework_of(args):
    """Name of the first recognized tensor framework among `args`, or 'numpy'."""
    for arg in args:
        if is_torch(arg):
            return "torch"
        if is_jax(arg):
            return "jax"
        if is_tf(arg):
            return "tensorflow"
    return "numpy"


def _compile(fn, name):
    """Compile `fn` for one framework; numpy (and anything unrecognized) stays uncompiled."""
    if name == "jax":
        import jax

        return jax.jit(fn)
    if name == "tensorflow":
        import tensorflow as tf

        return tf.function(fn, jit_compile=True)
    if name == "torch":
        import torch

        return torch.compile(fn)
    return fn
