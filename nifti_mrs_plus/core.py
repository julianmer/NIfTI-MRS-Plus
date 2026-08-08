####################################################################################################
#                                            core.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Purpose: NIfTI-MRS+ wrapper for efficient multi-subject (batched) processing across backends.    #
#                                                                                                  #
#   Wraps a list of "NIFTI_MRS" objects into a single batched representation with a shared         #
#   header, lazy per-backend tensor caching, and intuitive indexing.  Supports NIFTI_LIST,         #
#   NumPy, PyTorch, TensorFlow, JAX, and Keras backends so the same code seamlessly becomes        #
#   whatever tensor type the surrounding pipeline needs.                                           #
#                                                                                                  #
####################################################################################################

import importlib.util

import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from copy import deepcopy
import warnings

from nifti_mrs_plus import __version__

# Whether a framework is installed, answered without importing it. find_spec
# locates a module but does not execute it, so importing this package costs
# nothing even when torch, tensorflow, jax and keras are all present -- they are
# imported only by the conversion that actually needs one.
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None

from nifti_mrs.nifti_mrs import NIFTI_MRS


#**************************************************************************************************#
#                                         Class Backend                                            #
#**************************************************************************************************#
#                                                                                                  #
# Enumeration of the supported data/tensor backends.                                               #
#                                                                                                  #
#**************************************************************************************************#
class Backend(Enum):
    """Supported data backends."""
    NIFTI_LIST = "nifti_list"  # List of NIFTI_MRS objects
    NUMPY = "numpy"             # NumPy arrays
    PYTORCH = "pytorch"         # PyTorch tensors
    TENSORFLOW = "tensorflow"   # TensorFlow tensors
    JAX = "jax"                 # JAX arrays
    KERAS = "keras"             # Keras tensors (backed by TF/JAX)


#**************************************************************************************************#
#                                        Class DataState                                           #
#**************************************************************************************************#
#                                                                                                  #
# Where the data currently is, and what was last done to it.                                       #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class DataState:
    """
    Where the data currently is, and what was last done to it.

    Separate from the processing provenance, and deliberately so: provenance is
    a record that *grows* with every operation, which is why writing it is
    skipped in volatile mode. This is a fixed handful of facts, replaced rather
    than appended to, so it costs the same whether a pipeline has two steps or
    two hundred, and can be kept even on the fast path.

    A step reads it to work out what it is looking at. Which spectral and
    spatial domain the data is in decides whether a transform is needed;
    whether the data has been undersampled decides where noise may legitimately
    be added.

    Attributes:
        spectral: "time" or "frequency" - the state of the spectral axis.
        spatial: "image" or "kspace" - the state of the spatial axes. Kept
            apart from *spectral* because the two transforms act on different
            axes and commute, so the data can be in any combination of the two.
        sampling: "full" or "undersampled".
        last: Name of the operation that most recently touched the data.
    """

    spectral: str = 'time'
    spatial: str = 'image'
    sampling: str = 'full'
    last: str = ''

    def having(self, **changes) -> 'DataState':
        """
        A copy with *changes* applied, leaving this one untouched.

        Args:
            **changes: Any of the attributes above.

        Returns:
            The updated state.
        """
        return replace(self, **changes)


#****************#
#   provenance   #
#****************#
# NIfTI-MRS provenance records which software touched the data. This package is
# usually a library inside someone else's pipeline, so crediting itself would
# name the transport rather than the tool that did the work. Callers claim the
# record with set_provenance(); the default is honest for standalone use.
_PROVENANCE = {'program': 'nifti-mrs-plus', 'version': __version__}


def set_provenance(program: str, version: str):
    """Name the software that "ProcessingApplied" entries should credit.

    Call once at import time from the application or library built on top of
    this package::

        import nifti_mrs_plus
        nifti_mrs_plus.set_provenance('Augmentrum', augmentrum.__version__)

    Args:
        program: Software name written to the "Program" provenance field.
        version: Version string written to the "Version" provenance field.
    """
    _PROVENANCE['program'] = program
    _PROVENANCE['version'] = version


def get_provenance() -> dict:
    """Return the currently registered provenance "{'program', 'version'}"."""
    return dict(_PROVENANCE)


#**************************************************************************************************#
#                                      Class NIfTI_MRS_Plus                                        #
#**************************************************************************************************#
#                                                                                                  #
# Batch-aware wrapper around a list of NIFTI_MRS objects with multi-backend tensor access.         #
#                                                                                                  #
#**************************************************************************************************#
class NIfTI_MRS_Plus:
    """
    Handles batch-aware NIfTI_MRS data.

    This class wraps multiple 'NIFTI_MRS' objects into a batched representation.
    It provides access to the full batched tensor, header synchronization, and intuitive indexing.
    Supports multiple backends for different processing pipelines.

    Attributes:
        nifti_list (List[NIFTI_MRS]): List of individual subject MRS objects.
        backend (Backend): Backend for data representation (NIFTI_LIST, NumPy, PyTorch, etc.)
        volatile (bool): If True, skip metadata updates for speed

    Raises:
        TypeError: If any element in the list is not a NIFTI_MRS instance.
        ValueError: If all subjects don't have the same dim_tags.
    """

    def __init__(
        self,
        nifti_list: Union[List[NIFTI_MRS], 'NIfTI_MRS_Plus'],
        backend: Optional[Backend] = None,
        volatile: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        state: Optional[DataState] = None
    ):
        """
        Initialize the NIfTI_MRS_Plus object with data and header.

        Args:
            nifti_list: List of NIFTI_MRS objects or another NIfTI_MRS_Plus
            backend: Desired backend for pipeline processing (default: NIFTI_LIST)
            volatile: If True, skip metadata updates for speed
            metadata: Optional metadata dictionary
            state: Where the data is and what was last done to it. Carried from
                whatever produced it; a fresh batch starts at the NIfTI-MRS
                canonical form, time domain and image space.
        """
        self.volatile = volatile

        # Convert backend to enum if it's a string
        if isinstance(backend, str):
            backend = Backend[backend.upper()]

        self._backend = backend or Backend.NIFTI_LIST

        # Always store as list of NIfTI-MRS internally
        if isinstance(nifti_list, NIfTI_MRS_Plus):
            self.nifti_list = nifti_list.nifti_list
            if not volatile:
                self.metadata_common = deepcopy(nifti_list.metadata_common)
                self.metadata_individual = deepcopy(nifti_list.metadata_individual)
            else:
                self.metadata_common = {}
                self.metadata_individual = []
        elif isinstance(nifti_list, list) and len(nifti_list) > 0:
            # Check if list contains NIFTI-MRS-like objects (check for key attributes)
            first = nifti_list[0]
            if not (hasattr(first, 'shape') and hasattr(first, 'dwelltime')):
                error_msg = f"Data must be a list of NIFTI_MRS objects or NIfTI_MRS_Plus, got list of {type(first).__name__}"
                raise ValueError(error_msg)

            # Warn (not error) if subjects have different dim_tags.
            # NIFTI_LIST backend processes subjects individually so mixed
            # dim_tags is valid there.  Tensor backends will fail later in
            # numpy() with a clear "non-uniform shapes" message if stacking
            # is attempted.
            first_dim_tags = first.dim_tags if hasattr(first, 'dim_tags') else None
            for i, nifti_obj in enumerate(nifti_list[1:], 1):
                obj_dim_tags = nifti_obj.dim_tags if hasattr(nifti_obj, 'dim_tags') else None
                if obj_dim_tags != first_dim_tags:
                    warnings.warn(
                        f"NIfTI_MRS_Plus: subjects have non-uniform dim_tags "
                        f"(subject 0: {first_dim_tags}, subject {i}: {obj_dim_tags}). "
                        f"This is fine for Backend.NIFTI_LIST (subjects processed "
                        f"individually), but will fail if you switch to a tensor backend.",
                        stacklevel=2,
                    )
                    break  # one warning is enough

            self.nifti_list = nifti_list
            if not volatile:
                self._init_metadata(nifti_list, metadata)
            else:
                self.metadata_common = {}
                self.metadata_individual = []
        elif isinstance(nifti_list, list) and len(nifti_list) == 0:
            # Empty list - create empty NIfTI_MRS_Plus
            self.nifti_list = []
            self.metadata_common = {}
            self.metadata_individual = []
        else:
            error_msg = f"Data must be a list of NIFTI_MRS objects or NIfTI_MRS_Plus, got {type(nifti_list)}"
            if isinstance(nifti_list, list):
                if len(nifti_list) == 0:
                    error_msg += " (empty list)"
                else:
                    error_msg += f" (list of {type(nifti_list[0]).__name__})"
            raise ValueError(error_msg)

        # Track shape
        self.n_subjects = len(self.nifti_list)
        self._shape = (self.n_subjects,) + self.nifti_list[0].shape if self.n_subjects > 0 else (0,)

        # Single cache for backend-specific tensor
        # Stores tensor in the target backend (numpy/pytorch/etc), potentially on GPU
        self._cached_tensor = None
        self._cache_backend = None  # Which backend the cached tensor is in (relevant due to conversions)
        self._cache_device = None   # Device info (for PyTorch)
        # True when _cached_tensor holds newer values than nifti_list, i.e. it is
        # the source of truth and nifti_list is stale until materialize() runs.
        self._tensor_dirty = False
        # Kept even when volatile: it is a fixed handful of facts replaced each
        # step, not a record that grows, so it costs nothing to carry.
        self._state = state if state is not None else DataState()

        # Higher-dimension tags for the pending tensor, when they differ from
        # what nifti_list holds. Absolute, not a delta, so they survive being
        # handed from one processing step to the next.
        self._pending_tags = ()

    def _init_metadata(self, original_data: List[NIFTI_MRS], metadata: Optional[Dict] = None):
        """Initialize metadata from original NIfTI-MRS objects or provided dict."""
        if metadata is not None:
            self.metadata_common = metadata.get('common', {})
            self.metadata_individual = metadata.get('individual', [])
            return

        # Extract common and individual metadata from NIFTI_MRS objects
        self.metadata_common = {}
        self.metadata_individual = []

        # Common metadata (shared across all subjects)
        if len(original_data) > 0:
            first = original_data[0]
            self.metadata_common['dim_tags'] = first.dim_tags if hasattr(first, 'dim_tags') else []
            self.metadata_common['dwelltime'] = first.dwelltime if hasattr(first, 'dwelltime') else None
            self.metadata_common['spectrometer_frequency'] = first.spectrometer_frequency if hasattr(first, 'spectrometer_frequency') else None
            self.metadata_common['nucleus'] = first.nucleus if hasattr(first, 'nucleus') else None

            # Check if hdr_ext exists and extract common fields
            if hasattr(first, 'hdr_ext') and first.hdr_ext is not None:
                if 'EchoTime' in first.hdr_ext:
                    self.metadata_common['EchoTime'] = first.hdr_ext['EchoTime']
                if 'RepetitionTime' in first.hdr_ext:
                    self.metadata_common['RepetitionTime'] = first.hdr_ext['RepetitionTime']

        # Individual metadata (specific to each subject)
        for nifti_obj in original_data:
            individual_meta = {}
            if hasattr(nifti_obj, 'hdr_ext') and nifti_obj.hdr_ext is not None:
                if 'ProcessingApplied' in nifti_obj.hdr_ext:
                    individual_meta['ProcessingApplied'] = deepcopy(nifti_obj.hdr_ext['ProcessingApplied'])
                if 'ProcessingProvenance' in nifti_obj.hdr_ext:
                    individual_meta['ProcessingProvenance'] = deepcopy(nifti_obj.hdr_ext['ProcessingProvenance'])
            self.metadata_individual.append(individual_meta)

    @property
    def state(self) -> DataState:
        """Where the data is, and what was last done to it."""
        return self._state

    def set_state(self, state: DataState) -> 'NIfTI_MRS_Plus':
        """
        Record where the data now is.

        Args:
            state: The new state.

        Returns:
            "self", so calls can be chained.
        """
        self._state = state
        return self

    @property
    def backend(self) -> Backend:
        """Current backend setting."""
        return self._backend

    @property
    def shape(self) -> tuple:
        """Return shape of the batched data (n_subjects, ...)."""
        return self._shape

    @property
    def dim_tags(self):
        """
        Tags for the higher dimensions, as the data stands now.

        All subjects are validated to carry the same tags at construction, so
        the first speaks for the batch. Any dimension a pending tensor added is
        included, because the tags describe the data rather than the objects
        that are waiting to be rebuilt around it.
        """
        if self.n_subjects == 0:
            return None

        if self._pending_tags:
            return list(self._pending_tags)
        return list(self.nifti_list[0].dim_tags)

    @property
    def data(self):
        """
        Return data in the current backend format.
        For backward compatibility and pipeline use.
        """
        return self.get_data()

    def update_metadata(self, operation: str, details: Dict[str, Any],
                       individual_idx: Optional[List[int]] = None):
        """
        Update metadata with processing information.

        Args:
            operation: Name of operation applied
            details: Details of the operation
            individual_idx: If provided, update only these indices (for sample-specific ops)
        """
        if self.volatile:
            return  # Skip metadata updates in volatile mode

        from datetime import datetime

        provenance = {
            'Timestamp': datetime.now().isoformat(),
            'Operation': operation,
            'Details': str(details)
        }

        if individual_idx is None:
            # Common operation applied to all subjects
            if 'common_provenance' not in self.metadata_common:
                self.metadata_common['common_provenance'] = []
            self.metadata_common['common_provenance'].append(provenance)

            # Also update the actual NIfTI-MRS header extension for all subjects
            for nifti in self.nifti_list:
                if hasattr(nifti, 'hdr_ext') and nifti.hdr_ext is not None:
                    # Format for NIfTI-MRS provenance compatibility
                    processing_entry = {
                        'Time': provenance['Timestamp'],
                        'Program': _PROVENANCE['program'],
                        'Version': _PROVENANCE['version'],
                        'Method': operation,
                        'Details': str(details)
                    }

                    # Get existing ProcessingApplied list
                    if 'ProcessingApplied' in nifti.hdr_ext:
                        current_processing = nifti.hdr_ext['ProcessingApplied']
                    else:
                        current_processing = []

                    # Append new entry
                    current_processing.append(processing_entry)

                    # Use add_hdr_field to properly update the header extension
                    nifti.add_hdr_field('ProcessingApplied', current_processing)

        else:
            # Individual operations
            for idx in individual_idx:
                if idx < len(self.metadata_individual):
                    if 'ProcessingProvenance' not in self.metadata_individual[idx]:
                        self.metadata_individual[idx]['ProcessingProvenance'] = []
                    self.metadata_individual[idx]['ProcessingProvenance'].append(provenance)

                    # Also update the actual NIfTI-MRS header extension
                    if idx < len(self.nifti_list):
                        nifti = self.nifti_list[idx]
                        if hasattr(nifti, 'hdr_ext') and nifti.hdr_ext is not None:
                            try:
                                processing_entry = {
                                    'Time': provenance['Timestamp'],
                                    'Program': _PROVENANCE['program'],
                                    'Version': _PROVENANCE['version'],
                                    'Method': operation,
                                    'Details': str(details)
                                }

                                # Get existing ProcessingApplied list
                                if 'ProcessingApplied' in nifti.hdr_ext:
                                    current_processing = nifti.hdr_ext['ProcessingApplied']
                                else:
                                    current_processing = []

                                # Append new entry
                                current_processing.append(processing_entry)

                                # Use add_hdr_field to properly update the header extension
                                nifti.add_hdr_field('ProcessingApplied', current_processing)
                            except (TypeError, AttributeError):
                                # If header extension doesn't support modification, skip silently
                                pass

    def _ensure_individual(self) -> None:
        """Grow metadata_individual to one entry per spectrum.

        It is empty in volatile mode, and short whenever a caller passed a metadata
        dictionary that did not cover every spectrum.
        """
        while len(self.metadata_individual) < len(self.nifti_list):
            self.metadata_individual.append({})

    def set_result(self, name: str, value: Any, index: Optional[int] = None):
        """
        Attach an analysis result: a fit, concentrations, a QC metric.

        Kept apart from update_metadata, which records provenance and stringifies its
        details. A result has to come back out as the object that went in, or nothing
        downstream can compare it against anything. Results are stored even in volatile
        mode, where provenance is skipped: dropping a fit is not a speed optimisation.

        Args:
            name: What produced it, e.g. 'fit' or 'op_rmbadaverages'.
            value: The result itself. Any object; it is not serialised.
            index: Which spectrum it belongs to. None stores it for the batch.
        """
        if index is None:
            self.metadata_common.setdefault('results', {})[name] = value
            return

        self._ensure_individual()
        self.metadata_individual[index].setdefault('results', {})[name] = value

    def get_result(self, name: str, index: Optional[int] = None) -> Any:
        """
        Retrieve an attached result, or None if there is none by that name.

        Args:
            name: The name it was stored under.
            index: Which spectrum to read. None reads the batch-level result.

        Returns:
            The result as it was stored.
        """
        if index is None:
            return self.metadata_common.get('results', {}).get(name)

        self._ensure_individual()
        return self.metadata_individual[index].get('results', {}).get(name)

    def results(self, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Every result attached at this level.

        Args:
            index: Which spectrum to read. None reads the batch level.

        Returns:
            Dict[str, Any]: Results by name. Empty if none were attached.
        """
        if index is None:
            return dict(self.metadata_common.get('results', {}))

        self._ensure_individual()
        return dict(self.metadata_individual[index].get('results', {}))

    def copy(self) -> 'NIfTI_MRS_Plus':
        """Create a deep copy."""
        self.materialize()   # otherwise the copy is taken from stale values
        new_nifti_list = [nifti.copy() for nifti in self.nifti_list]

        return NIfTI_MRS_Plus(
            nifti_list=new_nifti_list,
            backend=self._backend,
            volatile=self.volatile,
            metadata={'common': deepcopy(self.metadata_common),
                     'individual': deepcopy(self.metadata_individual)},
            state=self._state
        )

    def numpy(self) -> np.ndarray:
        """
        Return batched tensor of shape [B, ...] where B = number of subjects.
        Caches the result for efficiency.

        Note:
            All subjects **must** have the same shape (same N_PTS and same
            extra dimensions).  If subjects have non-uniform shapes (e.g. after
            truncation to different lengths, or different numbers of coils/
            averages), this will raise a clear "ValueError".  Use
            "Backend.NIFTI_LIST" to process subjects individually in that case.
        """
        # Check if we have a cached numpy array
        if self._cached_tensor is not None and self._cache_backend == Backend.NUMPY:
            return self._cached_tensor

        # A pending tensor is newer than nifti_list, so restacking the list here
        # would silently return stale values. Convert the tensor instead.
        if self._tensor_dirty and self._cached_tensor is not None:
            from nifti_mrs_plus import ops
            return ops.to_numpy(self._cached_tensor)

        # Convert from nifti_list
        if len(self.nifti_list) == 0:
            arr = np.array([])
        else:
            try:
                arr = np.stack([n[:] for n in self.nifti_list], axis=0)
            except ValueError:
                shapes = [n[:].shape for n in self.nifti_list]
                unique = list(dict.fromkeys(shapes))   # preserve order, drop dupes
                raise ValueError(
                    f"NIfTI_MRS_Plus: cannot stack {len(self.nifti_list)} subjects into a "
                    f"single tensor - subjects have non-uniform shapes: {unique}.\n"
                    f"\n"
                    f"  Tensor backends (NumPy / PyTorch / TF / JAX / Keras) require every "
                    f"batch member to have identical shape.\n"
                    f"\n"
                    f"  Common causes:\n"
                    f"    - Apodization(mode='truncate') called with a different n_pts than the "
                    f"      batch was built with.\n"
                    f"    - A sampler / processor selected a different number of coils or "
                    f"      averages per subject.\n"
                    f"\n"
                    f"  Solutions:\n"
                    f"    - Keep N_PTS and all extra dimensions uniform across the batch "
                    f"      (fastest - single vectorised tensor op, in-place write-back).\n"
                    f"    - Or use Backend.NIFTI_LIST, which processes each subject "
                    f"      independently and handles any shape (slower, no batching)."
                ) from None

        # Cache it
        self._cached_tensor = arr
        self._cache_backend = Backend.NUMPY
        self._cache_device = 'cpu'

        return arr

    def list(self) -> List[NIFTI_MRS]:
        """
        Return list of NIFTI_MRS objects.

        Flushes any pending tensor first, so the objects returned always carry
        current values. That flush detaches: see "materialize".
        """
        self.materialize()
        return self.nifti_list

    #**************************#
    #   tensor-authoritative   #
    #**************************#
    def set_data(self, tensor, backend: Optional[Backend] = None, dim_tags=()):
        """
        Install *tensor* as the authoritative data, without writing to "nifti_list".

        This is the counterpart to "get_data", and what lets a processing
        pipeline preserve gradients: the tensor is kept exactly as handed over,
        "nifti_list" is marked stale, and the NumPy round-trip is deferred until
        something actually needs NIfTI objects.

        A pipeline therefore costs one materialization at the end, rather than
        one per step.

        Only the batch axis is fixed. A tensor of a different shape, or even a
        different rank, is accepted: the NIfTI objects are rebuilt to fit it at
        materialization, so an operation that resizes or adds a dimension is no
        less differentiable than one that does not.

        Args:
            tensor: Any backend tensor whose leading axis is the batch.
            backend: Which backend *tensor* belongs to. Inferred from its type
                when omitted.
            dim_tags: Higher-dimension tags for *tensor*, as a full
                "[dim_5, dim_6, dim_7]" list. Required when the rank grows,
                since a rebuilt object cannot infer what a new axis is.

        Returns:
            "self", so calls can be chained.

        Raises:
            ValueError: If the leading axis does not match "n_subjects".
        """
        if backend is None:
            backend = self._infer_backend(tensor)

        n = tensor.shape[0] if hasattr(tensor, 'shape') and len(tensor.shape) else None
        if n is not None and int(n) != self.n_subjects:
            raise ValueError(
                f"set_data: leading axis is {int(n)} but this batch holds "
                f"{self.n_subjects} subjects. The batch axis must be preserved."
            )

        self._cached_tensor = tensor
        self._cache_backend = backend
        self._cache_device = getattr(tensor, 'device', None) if backend == Backend.PYTORCH else None
        self._tensor_dirty = True
        self._pending_tags = tuple(dim_tags or ())
        return self

    _warned_jax_precision = False

    @classmethod
    def _warn_jax_precision(cls, dtype):
        """Warn once that JAX will silently halve the precision of 64-bit data."""
        try:
            import jax
            if jax.config.jax_enable_x64:
                return
        except ImportError:
            return

        if cls._warned_jax_precision:
            return
        cls._warned_jax_precision = True
        warnings.warn(
            f"JAX will downcast {dtype} to its 32-bit counterpart because "
            "jax_enable_x64 is off, so this batch carries less precision on JAX "
            "than on NumPy or TensorFlow. Pass dtype='complex64' to get_data() to "
            "make that explicit and consistent, or enable x64 with "
            "jax.config.update('jax_enable_x64', True).",
            RuntimeWarning,
            stacklevel=3,
        )

    @staticmethod
    def _infer_backend(tensor) -> Backend:
        """Map a tensor to the Backend it belongs to, by module name."""
        from nifti_mrs_plus import ops

        if isinstance(tensor, np.ndarray):
            return Backend.NUMPY
        if ops.is_torch(tensor):
            return Backend.PYTORCH
        if ops.is_jax(tensor):
            return Backend.JAX
        if ops.is_tf(tensor):
            return Backend.TENSORFLOW
        raise TypeError(
            f"set_data: cannot infer a backend for {type(tensor).__name__}. "
            "Pass backend= explicitly."
        )

    def materialize(self):
        """
        Write any pending tensor back into "nifti_list".

        This is the single point at which data leaves its framework and becomes
        NumPy again, so it is also the single point at which an autograd graph
        ends and a device tensor returns to the host. Everything needing real
        "NIFTI_MRS" objects -- "list", "save_nifti",
        "save_hdf5" -- goes through here.

        A no-op when nothing is pending, so it is cheap to call defensively.

        A sample whose shape no longer matches its object gets a rebuilt object,
        because a NIFTI_MRS fixes its extent at construction. Deferring that to
        here is what lets a resizing operation stay differentiable: the rebuild
        happens once, at the end, rather than at every step that changed a shape.

        Returns:
            "self", so calls can be chained.
        """
        if not self._tensor_dirty or self._cached_tensor is None:
            return self

        from nifti_mrs_plus import ops

        arr = ops.to_numpy(self._cached_tensor)
        for i, nifti in enumerate(self.nifti_list):
            if i >= arr.shape[0]:
                break
            if arr[i].shape == nifti[:].shape:
                nifti[:] = arr[i]
            else:
                self.nifti_list[i] = self._rebuild(nifti, arr[i])

        self._tensor_dirty = False
        return self

    def _rebuild(self, nifti: NIFTI_MRS, sample: np.ndarray) -> NIFTI_MRS:
        """
        A NIFTI_MRS holding *sample*, carrying *nifti*'s header across.

        Args:
            nifti: Object whose header and geometry to keep.
            sample: Values for the new object, one subject's worth.

        Returns:
            The rebuilt object.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext

        hdr_ext = nifti.hdr_ext.copy()

        # Only dimensions the new sample actually has can carry a tag.
        tags = list(self._pending_tags) if self._pending_tags else list(nifti.dim_tags)
        for position in range(3):
            wanted = tags[position] if position < max(0, sample.ndim - 4) else None
            hdr_ext.set_dim_info(position, wanted)

        try:
            affine = nifti.getAffine('voxel', 'world')
        except Exception:
            affine = None

        rebuilt = gen_nifti_mrs_hdr_ext(sample, nifti.dwelltime, hdr_ext, affine=affine)

        # Callers subclass NIFTI_MRS to add their own methods -- FSL-MRS is one --
        # and the rebuilt object stands in for the original everywhere the
        # original went, so it has to be the same kind of thing.
        if type(rebuilt) is not type(nifti):
            rebuilt = type(nifti)(rebuilt)
        return rebuilt

    def _invalidate_cache(self):
        """Invalidate the cached tensor (called when data is modified)."""
        self._cached_tensor = None
        self._cache_backend = None
        self._cache_device = None
        self._tensor_dirty = False
        self._pending_tags = ()

    def get_data(self, backend: Optional[Backend] = None, dtype=None):
        """
        Get data in the specified backend format.

        Efficiently caches tensors in the target backend (including GPU tensors).
        Only converts when necessary.

        Args:
            backend: Desired backend (uses instance backend if None)
            dtype: NumPy dtype to cast to before conversion, e.g. "'complex64'".
                Defaults to "None", which preserves whatever the NIfTI data
                holds. Pass it explicitly to pin precision across backends --
                JAX downcasts 64-bit types unless "jax_enable_x64" is set,
                so the same file otherwise yields complex64 on JAX and
                complex128 on NumPy and TensorFlow.

        Returns:
            Data in requested format (list of NIFTI_MRS, numpy array, or tensor)
        """
        target = backend or self._backend

        if target == Backend.NIFTI_LIST:
            self.materialize()
            return self.nifti_list

        # Check if we already have cached tensor in target backend
        if dtype is None and self._cached_tensor is not None and self._cache_backend == target:
            return self._cached_tensor

        # Get numpy array as intermediate (cached)
        array = self.numpy()

        if dtype is not None:
            array = array.astype(dtype)
        elif target == Backend.JAX and array.dtype.itemsize > 8:
            self._warn_jax_precision(array.dtype)

        if target == Backend.NUMPY:
            return array  # Already cached by numpy()

        tensor = self._convert(array, target)

        # An explicitly requested dtype is a one-off view. Caching it would make
        # a later default call return that dtype instead of the stored one.
        if dtype is None:
            self._cached_tensor = tensor
            self._cache_backend = target
            self._cache_device = 'cpu' if target == Backend.PYTORCH else None

        return tensor

    @staticmethod
    def _convert(array: np.ndarray, target: Backend):
        """Convert a NumPy array to *target*'s tensor type, without caching."""
        if target == Backend.PYTORCH:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            import torch
            # torch.from_numpy shares memory with `array`, which is this object's
            # cached NumPy view. An in-place torch op would then mutate the cache
            # behind its own back, so take a copy.
            return torch.from_numpy(array.copy())

        if target == Backend.TENSORFLOW:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")
            import tensorflow as tf
            return tf.convert_to_tensor(array)

        if target == Backend.JAX:
            if not JAX_AVAILABLE:
                raise ImportError("JAX not available")
            import jax.numpy as jnp
            return jnp.array(array)

        if target == Backend.KERAS:
            if not KERAS_AVAILABLE:
                raise ImportError("Keras not available")
            import keras.ops as keras_ops
            return keras_ops.convert_to_tensor(array)

        raise ValueError(f"Unknown backend: {target}")

    def to(self, device: str) -> 'NIfTI_MRS_Plus':
        """
        Move data to specified device (for PyTorch backend).

        Args:
            device: Device string ('cuda', 'cpu', 'cuda:0', etc.)

        Returns:
            New NIfTI_MRS_Plus with data on the specified device

        Example:
            >>> nifti_gpu = nifti_plus.to('cuda')
            >>> nifti_cpu = nifti_gpu.to('cpu')
        """
        if self._backend != Backend.PYTORCH:
            raise ValueError(f"`.to(device)` only works with PyTorch backend, current backend is {self._backend.value}")

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        # Get current tensor
        tensor = self.get_data(Backend.PYTORCH)

        # Move to device
        tensor_on_device = tensor.to(device)

        # Create new NIfTI_MRS_Plus (reuse nifti_list, new tensor cache)
        new_obj = NIfTI_MRS_Plus(
            nifti_list=self.nifti_list,  # Reuse same nifti objects
            backend=Backend.PYTORCH,
            volatile=self.volatile,
            metadata={'common': self.metadata_common, 'individual': self.metadata_individual},
            state=self._state
        )

        # Set the unified cache with GPU tensor
        new_obj._cached_tensor = tensor_on_device
        new_obj._cache_backend = Backend.PYTORCH
        new_obj._cache_device = device

        return new_obj

    def to_nifti_list(self) -> List[NIFTI_MRS]:
        """Return the internal list of NIfTI-MRS objects."""
        self.materialize()
        return self.nifti_list

    # Proxy methods that delegate to first NIFTI_MRS (all have same structure)
    def dim_position(self, dim_tag: str) -> Optional[int]:
        """
        Get the position of a dimension tag.
        Delegates to first NIFTI_MRS (all subjects have same dim_tags).
        """
        if self.n_subjects > 0:
            return self.nifti_list[0].dim_position(dim_tag)
        return None

    def set_dim_tag(self, index: int, tag: str):
        """
        Set dimension tag for all subjects.
        Since all subjects must have same dimTags, this updates all.
        """
        for nifti in self.nifti_list:
            nifti.set_dim_tag(index, tag)
        # Update cached common metadata
        if not self.volatile and self.n_subjects > 0:
            self.metadata_common['dim_tags'] = self.nifti_list[0].dim_tags

    def sync_headers(self, source_idx: int = 0) -> None:
        """
        Sync headers from one subject to all others.

        Updates all headers to match the header of the subject at 'source_idx'.

        Args:
            source_idx (int): Index of the reference subject.
        """
        if self.n_subjects == 0:
            return

        ref_hdr = deepcopy(self.nifti_list[source_idx].hdr_ext)
        for nifti in self.nifti_list:
            nifti.hdr_ext = deepcopy(ref_hdr)

        # Update cached metadata
        if not self.volatile:
            self.metadata_common['hdr_ext'] = ref_hdr

    @property
    def dwelltime(self):
        """Dwelltime from metadata (all subjects have same value)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].dwelltime
        return self.metadata_common.get('dwelltime')

    @property
    def spectrometer_frequency(self):
        """Spectrometer frequency from metadata (all subjects have same value)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].spectrometer_frequency
        return self.metadata_common.get('spectrometer_frequency')

    @property
    def nucleus(self):
        """Nucleus from metadata (all subjects have same value)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].nucleus
        return self.metadata_common.get('nucleus')

    @property
    def ndim(self):
        """Number of dimensions (from header extension)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].ndim
        return None

    @property
    def dtype(self):
        """Data type of the underlying data."""
        if self.n_subjects > 0:
            return self.nifti_list[0].dtype
        return None

    @property
    def header(self):
        """Header from first NIFTI_MRS (all have same structure)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].header
        return None

    @property
    def hdr_ext(self):
        """Header extension from first NIFTI_MRS (all have same structure)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].hdr_ext
        return None

    def __getitem__(self, idx):
        """
        Get subject(s) from the batch.

        Args:
            idx: Integer index or slice

        Returns:
            - If idx is int: Returns the NIFTI_MRS object directly
            - If idx is slice: Returns a new NIfTI_MRS_Plus with subset of subjects

        Examples:
            >>> nifti_plus[0]  # Returns NIFTI_MRS object for first subject
            >>> nifti_plus[0:5]  # Returns NIfTI_MRS_Plus with subjects 0-4
        """
        # Hands out real NIFTI_MRS objects, so a pending tensor has to land
        # first or the caller reads stale values.
        self.materialize()

        if isinstance(idx, int):
            # Single index - return the NIFTI_MRS object directly
            return self.nifti_list[idx]
        else:
            # Slice - return new NIfTI_MRS_Plus with subset
            new_nifti_list = self.nifti_list[idx]
            new_individual = self.metadata_individual[idx] if not self.volatile else []

            return NIfTI_MRS_Plus(
                nifti_list=new_nifti_list,
                backend=self._backend,
                volatile=self.volatile,
                metadata={'common': self.metadata_common, 'individual': new_individual},
                state=self._state
            )

    def __setitem__(self, idx, values):
        """
        Set data values.

        Supports setting values for all backends. When setting values, the cached
        data is invalidated to ensure consistency.

        Args:
            idx: Index to set (can be int, slice, tuple for multi-dimensional indexing)
            values: Values to set (can be array, tensor, or list depending on backend)

        Examples:
            # For NIFTI_LIST backend
            nifti_plus[0] = new_nifti_mrs  # Set entire subject

            # For array/tensor backends
            nifti_plus[:, :, :, :, 100:200] = new_values  # Set specific timepoints
        """
        if self._backend == Backend.NIFTI_LIST:
            # Setting in NIFTI_LIST backend
            if isinstance(idx, (int, slice)):
                # Subject-level assignment
                if isinstance(values, list):
                    # Assigning list of NIFTI_MRS to multiple subjects
                    if isinstance(idx, int):
                        self.nifti_list[idx] = values[0] if len(values) == 1 else values
                    else:
                        self.nifti_list[idx] = values
                elif hasattr(values, 'dwelltime'):
                    # Assigning single NIFTI_MRS
                    self.nifti_list[idx] = values
                else:
                    raise ValueError(f"For NIFTI_LIST backend, values must be NIFTI_MRS object(s), got {type(values)}")
            else:
                # Multi-dimensional indexing - set data within each NIFTI_MRS
                if isinstance(values, list):
                    for i, nifti in enumerate(self.nifti_list):
                        nifti[idx] = values[i]
                else:
                    # Broadcast same values to all subjects
                    for nifti in self.nifti_list:
                        nifti[idx] = values

            # Invalidate cache
            self._invalidate_cache()

        else:
            # Setting in array/tensor backends
            # Need to update both the cached array and the underlying NIFTI objects

            from nifti_mrs_plus import ops

            # Whole-batch assignment is the pipeline's hot path, and demoting it
            # to NumPy here would sever an autograd graph and pull a device
            # tensor back to the host. Hand it to set_data instead, which keeps
            # the tensor as-is and defers the round-trip to materialize().
            if isinstance(idx, slice) and idx == slice(None) and not isinstance(values, list):
                try:
                    self.set_data(values)
                except TypeError:
                    pass          # not a recognized tensor; fall through to NumPy
                else:
                    return

            # Partial assignment still goes through NumPy: writing into a slice
            # of a NIfTI array cannot preserve a graph. to_numpy detaches
            # explicitly, rather than raising the way .numpy() does on a
            # grad-enabled or non-CPU tensor.
            values_np = ops.to_numpy(values) if not isinstance(values, np.ndarray) else values

            # Update cached tensor if it exists
            if self._cached_tensor is not None:
                self._cached_tensor[idx] = values_np

            # Update underlying NIFTI objects
            # For batched indexing like nifti_plus[0, :, :, :, 100:200]
            if isinstance(idx, tuple) and len(idx) > 0:
                # First element is subject index
                if isinstance(idx[0], int):
                    # Single subject
                    subject_idx = idx[0]
                    data_idx = idx[1:] if len(idx) > 1 else (slice(None),)
                    self.nifti_list[subject_idx][data_idx] = values_np[0] if values_np.ndim > 0 and values_np.shape[0] == 1 else values_np
                elif isinstance(idx[0], slice):
                    # Multiple subjects
                    subject_slice = idx[0]
                    data_idx = idx[1:] if len(idx) > 1 else (slice(None),)
                    subjects = self.nifti_list[subject_slice]
                    for i, nifti in enumerate(subjects):
                        # Handle broadcasting: if values_np has only 1 subject, broadcast to all
                        if values_np.shape[0] == 1:
                            nifti[data_idx] = values_np[0]
                        elif i < len(values_np):
                            nifti[data_idx] = values_np[i]
                        else:
                            raise ValueError(f"Not enough values to assign: need {len(subjects)}, got {len(values_np)}")
                else:
                    # Array indexing
                    raise NotImplementedError("Advanced indexing with arrays not yet supported for __setitem__")
            else:
                # Simple indexing - update all underlying NIFTI objects
                for i, nifti in enumerate(self.nifti_list):
                    nifti[idx] = values_np[i] if values_np.ndim > 0 and len(values_np) > i else values_np

    def __len__(self) -> int:
        """Number of subjects."""
        return self.n_subjects

    def __repr__(self) -> str:
        return (f"NIfTI_MRS_Plus(n_subjects={self.n_subjects}, shape={self.shape}, "
                f"backend={self._backend.value}, volatile={self.volatile})")

    # ===========================================================================================
    # PLOTTING METHODS
    # ===========================================================================================

    def plot(
        self,
        display_dim=None,
        ppmlim=None,
        plot_avg=False,
        batch_index=None,
        max_batch_display=6,
        grid_layout=None,
        legend=True,
        figsize=None,
        title=None,
        mask=None
    ):
        """
        Plot NIfTI_MRS_Plus spectra with batch-aware visualization.

        Mirrors the NIfTI-MRS plot() method but adds batch support:
        - Single batch element: plots like NIfTI-MRS
        - Multiple batch elements: grid or comparison plot

        Args:
            display_dim: Dimension to display (if multiple dims exist in individual spectra)
            ppmlim: Tuple of (min_ppm, max_ppm) for x-axis limits
            plot_avg: If True, plot average spectrum across dimensions
            batch_index: Which batch element to plot (None = plot multiple)
            max_batch_display: Maximum number of batch elements to display (default: 6)
            grid_layout: Tuple (rows, cols) for grid layout. Auto-calculated if None
            legend: Whether to show legend
            figsize: Figure size tuple (width, height)
            title: Plot title
            mask: Spatial mask (for MRSI data)

        Returns:
            matplotlib figure object

        Examples:
            >>> # Plot first spectrum (like NIfTI-MRS)
            >>> nifti_plus.plot(batch_index=0)

            >>> # Plot first 4 spectra in grid
            >>> nifti_plus.plot(max_batch_display=4)

            >>> # Plot with custom ppm range
            >>> nifti_plus.plot(ppmlim=(0.5, 4.2))

            >>> # Plot specific batch element with custom settings
            >>> nifti_plus.plot(batch_index=2, ppmlim=(1.0, 4.0), title="Augmented Spectrum")
        """
        from nifti_mrs_plus.plotting import vis_nifti_mrs_plus

        return vis_nifti_mrs_plus(
            self,
            display_dim=display_dim,
            ppmlim=ppmlim,
            plot_avg=plot_avg,
            batch_index=batch_index,
            max_batch_display=max_batch_display,
            grid_layout=grid_layout,
            legend=legend,
            figsize=figsize,
            title=title
        )

    def plot_comparison(
        self,
        indices=None,
        ppmlim=(0.2, 4.2),
        labels=None,
        title="Batch Comparison",
        colors=None,
        alpha=0.7,
        figsize=(12, 5)
    ):
        """
        Plot multiple batch spectra overlaid for comparison.

        Useful for comparing original vs augmented spectra.

        Args:
            indices: List of batch indices to plot (None = first 6)
            ppmlim: PPM range tuple (min, max)
            labels: List of labels for each spectrum
            title: Plot title
            colors: List of colors (auto if None)
            alpha: Transparency (0-1)
            figsize: Figure size

        Returns:
            matplotlib figure

        Example:
            >>> nifti_plus.plot_comparison(indices=[0, 1, 2],
            ...                           labels=['Original', 'Aug 1', 'Aug 2'])
        """
        from nifti_mrs_plus.plotting import plot_batch_comparison

        return plot_batch_comparison(
            self,
            indices=indices,
            ppmlim=ppmlim,
            labels=labels,
            title=title,
            colors=colors,
            alpha=alpha,
            figsize=figsize
        )

    def plot_grid(
        self,
        max_display=9,
        ppmlim=(0.2, 4.2),
        title="Batch Grid",
        show_metabolites=False,
        figsize=None
    ):
        """
        Plot batch elements in a detailed grid layout.

        Args:
            max_display: Maximum spectra to display (default: 9)
            ppmlim: PPM range
            title: Overall title
            show_metabolites: Highlight common metabolite regions (NAA, Cr, Cho)
            figsize: Figure size (auto if None)

        Returns:
            matplotlib figure

        Example:
            >>> nifti_plus.plot_grid(max_display=6, show_metabolites=True)
        """
        from nifti_mrs_plus.plotting import plot_batch_grid_detailed

        return plot_batch_grid_detailed(
            self,
            max_display=max_display,
            ppmlim=ppmlim,
            title=title,
            show_metabolites=show_metabolites,
            figsize=figsize
        )

    # ===========================================================================================
    # EXPORT METHODS
    # ===========================================================================================

    def save_nifti(self, output_dir: str, prefix: str = "spectrum", zero_pad: int = 4):
        """
        Save each subject as a separate NIFTI-MRS file.

        Args:
            output_dir: Directory to save NIFTI files
            prefix: Prefix for filenames (default: "spectrum")
            zero_pad: Zero-padding width for subject numbering (default: 4)

        Returns:
            List of saved file paths

        Example:
            >>> nifti_plus.save_nifti('output/', prefix='aug', zero_pad=3)
            ['output/aug_001.nii.gz', 'output/aug_002.nii.gz', ...]
        """
        from pathlib import Path

        self.materialize()   # a pending tensor would otherwise not reach disk

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for i, nifti_obj in enumerate(self.nifti_list):
            # Format filename with zero-padding
            filename = f"{prefix}_{i:0{zero_pad}d}.nii.gz"
            filepath = output_path / filename

            # Save using NIfTI-MRS save method
            nifti_obj.save(str(filepath))
            saved_files.append(str(filepath))

        return saved_files

    def save_hdf5(self, filepath: str, compression: str = 'gzip', compression_opts: int = 4):
        """
        Save NIfTI_MRS_Plus object to HDF5 file with full metadata preservation.

        This stores all subjects in a single HDF5 file with:
        - Batched data array
        - Common metadata (shared across subjects)
        - Individual metadata (per-subject provenance, processing history)
        - NIfTI headers

        Args:
            filepath: Path to save HDF5 file
            compression: Compression method ('gzip', 'lzf', None)
            compression_opts: Compression level (0-9 for gzip)

        Example:
            >>> nifti_plus.save_hdf5('dataset.h5')
            >>> # Later: load with NIfTI_MRS_Plus.load_hdf5('dataset.h5')
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 saving. Install with: pip install h5py")

        from pathlib import Path
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['n_subjects'] = self.n_subjects
            f.attrs['backend'] = self._backend.value
            f.attrs['volatile'] = self.volatile
            f.attrs['nifti_mrs_plus_version'] = __version__
            # Who wrote this file, per set_provenance(). Recorded generically so
            # a reader can identify the producing tool without this package
            # having to know the names of its downstream users.
            f.attrs['program'] = _PROVENANCE['program']
            f.attrs['program_version'] = _PROVENANCE['version']

            # Store batched data
            batched_data = self.numpy()
            f.create_dataset(
                'data',
                data=batched_data,
                compression=compression,
                compression_opts=compression_opts
            )

            # Store common metadata
            if self.metadata_common:
                metadata_grp = f.create_group('metadata_common')
                for key, value in self.metadata_common.items():
                    if value is not None:
                        if isinstance(value, (list, tuple)):
                            metadata_grp.attrs[key] = json.dumps(value)
                        else:
                            metadata_grp.attrs[key] = value

            # Store individual metadata
            if self.metadata_individual:
                individual_grp = f.create_group('metadata_individual')
                for i, meta in enumerate(self.metadata_individual):
                    if meta:
                        subject_grp = individual_grp.create_group(f'subject_{i:04d}')
                        for key, value in meta.items():
                            if isinstance(value, (dict, list)):
                                subject_grp.attrs[key] = json.dumps(value)
                            else:
                                subject_grp.attrs[key] = str(value)

            # Store NIfTI headers (serialized)
            headers_grp = f.create_group('nifti_headers')
            for i, nifti_obj in enumerate(self.nifti_list):
                subject_header_grp = headers_grp.create_group(f'subject_{i:04d}')

                # Store basic header info
                if hasattr(nifti_obj, 'header'):
                    try:
                        subject_header_grp.attrs['dim'] = json.dumps(list(nifti_obj.header['dim']))
                        subject_header_grp.attrs['pixdim'] = json.dumps(list(nifti_obj.header['pixdim']))
                    except Exception:
                        pass  # Skip if can't serialize

                # Store dwelltime and frequency
                if hasattr(nifti_obj, 'dwelltime'):
                    subject_header_grp.attrs['dwelltime'] = float(nifti_obj.dwelltime)
                if hasattr(nifti_obj, 'spectrometer_frequency'):
                    subject_header_grp.attrs['spectrometer_frequency'] = json.dumps(list(nifti_obj.spectrometer_frequency))

                # The NIfTI-MRS header extension *is* JSON by specification, and
                # upstream Hdr_Ext round-trips through it. Storing it is what
                # makes load_hdf5 able to rebuild real NIFTI_MRS objects.
                try:
                    subject_header_grp.attrs['hdr_ext'] = nifti_obj.hdr_ext.to_json()
                except Exception:
                    pass
                try:
                    affine = nifti_obj.getAffine('voxel', 'world')
                    subject_header_grp.attrs['affine'] = json.dumps(np.asarray(affine).tolist())
                except Exception:
                    pass

    @classmethod
    def load_hdf5(cls, filepath: str, backend: Optional[Backend] = None) -> 'NIfTI_MRS_Plus':
        """
        Load NIfTI_MRS_Plus object from HDF5 file.

        Args:
            filepath: Path to HDF5 file
            backend: Backend to use (default: same as saved)

        Returns:
            NIfTI_MRS_Plus object

        Example:
            >>> nifti_plus = NIfTI_MRS_Plus.load_hdf5('dataset.h5')
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 loading. Install with: pip install h5py")

        import json
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        with h5py.File(filepath, 'r') as f:
            # Load metadata
            n_subjects = f.attrs['n_subjects']
            saved_backend = Backend[f.attrs['backend'].upper()]
            volatile = bool(f.attrs.get('volatile', False))

            # Load data
            batched_data = f['data'][:]

            # Load common metadata
            metadata_common = {}
            if 'metadata_common' in f:
                for key, value in f['metadata_common'].attrs.items():
                    try:
                        metadata_common[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        metadata_common[key] = value

            # Load individual metadata
            metadata_individual = []
            if 'metadata_individual' in f:
                for i in range(n_subjects):
                    subject_key = f'subject_{i:04d}'
                    if subject_key in f['metadata_individual']:
                        meta = {}
                        for key, value in f['metadata_individual'][subject_key].attrs.items():
                            try:
                                meta[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                meta[key] = value
                        metadata_individual.append(meta)
                    else:
                        metadata_individual.append({})

            # Rebuild one NIFTI_MRS per subject from the stored header extension.
            if 'nifti_headers' not in f:
                raise ValueError(
                    f"{filepath} has no 'nifti_headers' group, so NIFTI_MRS objects cannot "
                    "be rebuilt. It was written by a version of nifti-mrs-plus that did not "
                    "store the header extension; re-save it with the current version."
                )

            nifti_list = []
            for i in range(int(n_subjects)):
                grp = f['nifti_headers'][f'subject_{i:04d}']

                if 'hdr_ext' not in grp.attrs:
                    raise ValueError(
                        f"{filepath}: subject {i} has no stored header extension. "
                        "Re-save with the current version of nifti-mrs-plus."
                    )

                hdr_ext = Hdr_Ext.from_header_ext(json.loads(grp.attrs['hdr_ext']))
                affine = (np.array(json.loads(grp.attrs['affine']))
                          if 'affine' in grp.attrs else None)

                nifti_list.append(
                    gen_nifti_mrs_hdr_ext(
                        batched_data[i],
                        float(grp.attrs['dwelltime']),
                        hdr_ext,
                        affine=affine,
                    )
                )

            return cls(
                nifti_list=nifti_list,
                backend=backend or saved_backend,
                volatile=volatile,
                metadata={'common': metadata_common, 'individual': metadata_individual},
            )
