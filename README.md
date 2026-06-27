<div align="center">
  <img src="https://raw.githubusercontent.com/julianmer/NIfTI-MRS-Plus/main/assets/logo.svg" alt="NIfTI-MRS+ Logo" width="200"/>
  <p><em>Efficient, backend-agnostic batching of NIfTI-MRS spectra</em></p>

  [![PyPI version](https://badge.fury.io/py/nifti-mrs-plus.svg)](https://badge.fury.io/py/nifti-mrs-plus)
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
</div>

---

`nifti-mrs-plus` wraps a list of [NIfTI-MRS](https://github.com/wtclarke/mrs_nifti_standard)
objects into a single batch-aware tensor, keeping the header, dimension tags, and provenance
intact. It materializes in whatever backend your pipeline needs (NumPy, PyTorch, JAX,
TensorFlow, or Keras) with no code changes.

---

## Install

```bash
pip install nifti-mrs-plus                   # core (numpy + nifti-mrs)
pip install "nifti-mrs-plus[torch]"          # + PyTorch
pip install "nifti-mrs-plus[viz]"            # + plotting (matplotlib + fsl-mrs)
pip install "nifti-mrs-plus[all]"            # everything
```

Extras: `torch` · `jax` · `tensorflow` · `keras` · `viz` · `hdf5` · `all` · `dev`

---

## Quick start

```python
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend

batch = NIfTI_MRS_Plus(nifti_list)

batch.shape                          # (8, 1, 1, 1, 2048)
batch.numpy()                        # np.ndarray, cached
batch.get_data(Backend.PYTORCH)      # torch.Tensor  (if torch installed)
batch[0]                             # NIFTI_MRS  (single subject)
batch[0:4]                           # NIfTI_MRS_Plus  (sub-batch)
```

---

## Backends

| Extra | `Backend` | Returns |
|---|---|---|
| *(none)* | `NIFTI_LIST` | `list[NIFTI_MRS]` |
| *(none)* | `NUMPY` | `numpy.ndarray` |
| `torch` | `PYTORCH` | `torch.Tensor` |
| `jax` | `JAX` | `jax.Array` |
| `tensorflow` | `TENSORFLOW` | `tf.Tensor` |
| `keras` | `KERAS` | Keras tensor |

---

<div align="center">
  <sub>Built with ❤️ for the MRS community</sub>
</div>