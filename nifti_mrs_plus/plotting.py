####################################################################################################
#                                           plotting.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-02-13                                                                              #
#                                                                                                  #
# Purpose: Batch-aware plotting utilities for NIfTI_MRS_Plus objects.                              #
#          Implements spectrum / grid / comparison plots following NIfTI-MRS conventions.          #
#          Coil combination and spectrum extraction use fsl-mrs, exactly as the reference          #
#          nifti-mrs visualisation does (fsl-mrs required for plotting).                           #
#                                                                                                  #
####################################################################################################

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

try:
    from fsl_mrs.utils.plotting import plot_spectrum, plot_spectra
    from fsl_mrs.core.mrs import MRS
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.utils import constants
except ImportError:
    raise ImportError(
        "NIfTI-MRS visualisation requires FSL-MRS tools to be installed. "
        "See fsl-mrs.com for installation instructions.")


def _process_to_mrs(nifti, ppmlim=None):
    """Reduce a NIFTI_MRS object to an fsl-mrs MRS object, mirroring nifti-mrs vis.

    Coils (DIM_COIL) are coil-combined and every other non-singleton dimension is
    averaged (or subtracted for edited acquisitions), just like the reference
    nifti-mrs visualisation. Returns an fsl_mrs MRS instance.
    """
    data = nifti

    if data.ndim > 4 \
            and 'DIM_COIL' in data.dim_tags \
            and data.shape[data.dim_position('DIM_COIL')] > 1:
        data = nifti_mrs_proc.coilcombine(data)

    for dim in data.dim_tags:
        if dim is None:
            continue
        if data.shape[data.dim_position(dim)] > 1:
            if dim in ('DIM_EDIT', 'DIM_METCYCLE', 'DIM_ISIS'):
                data = nifti_mrs_proc.subtract(data, dim=dim)
            else:
                data = nifti_mrs_proc.average(data, dim)

    return MRS(
        data[:].squeeze(),
        bw=data.bandwidth,
        cf=data.spectrometer_frequency[0],
        nucleus=data.nucleus[0])


def _spectrum(nifti, ppmlim=None):
    """Return (ppm, spec) for a NIFTI_MRS object using fsl-mrs."""
    mrs = _process_to_mrs(nifti, ppmlim=ppmlim)
    ppm = mrs.getAxes(ppmlim=ppmlim)
    spec = mrs.get_spec(ppmlim=ppmlim)
    return ppm, spec


def _default_ppmlim(nucleus):
    """Default ppm window for a nucleus, via fsl-mrs nucleus constants."""
    nuc_info = constants.nucleus_constants(nucleus)
    if nuc_info.ppm_range:
        return nuc_info.ppm_range
    return (0.2, 4.2)


# ══════════════════════════════════════════════════════════════════════════════
#  Simple spectrum / fit plots  (MRS paper style)
# ══════════════════════════════════════════════════════════════════════════════

def _crop_ppm(arr, ppmAxis, ppmLim):
    """Return (arr_cropped, ppmAxis_cropped) within ppmLim = (lo, hi)."""
    beg = int((np.abs(ppmAxis - ppmLim[0])).argmin())
    end = int((np.abs(ppmAxis - ppmLim[1])).argmin())
    if beg > end:
        beg, end = end, beg
    return arr[beg:end], ppmAxis[beg:end]


def _pick_part(arr, real):
    """Select real or imaginary part of a (possibly already real) array."""
    return np.real(arr) if real else np.imag(arr)


def _style_mrs_ax(ax, clean):
    """Canonical MRS axis style (inverted ppm, y-hidden); everything hidden when clean."""
    for spine in ('right', 'top', 'left'):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.invert_xaxis()
    if clean:
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks_position('none')


def _save_fig(fig, save_path, name):
    """Save fig to save_path/name.png, close it, and return the path."""
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f'{name}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
        return out
    return None


def plot_spec_and_fit(spec, fit, true=None, ppmAxis=None, ppmLim=None, name='',
                      save_path=None, real=True, clean=False):
    """Plot a measured spectrum, the model fit, and the residual on a ppm axis.

    Visual style: Data (black) / Fit (red, alpha=0.6) / Residual offset above (dimgray) /
    optional True spectrum (green dashed). ppm-axis inverted, y-axis hidden.

    Parameters
    ----------
    spec      : np.ndarray (complex)  measured spectrum (1-D).
    fit       : np.ndarray (complex)  forward-modeled fit (1-D).
    true      : np.ndarray (complex), optional  ground-truth spectrum.
    ppmAxis   : np.ndarray, optional  chemical-shift axis matching spec. Defaults to
                linspace(0.5, 4.0, len(spec)).
    ppmLim    : tuple(float, float), optional  (lo, hi) ppm window to display.
    name      : str  filename stem (without extension).
    save_path : str, optional  directory to save PNG into; created if missing.
    real      : bool  if True plot real part, else imaginary.
    clean     : bool  if True hide all axes/labels (publication snippet style).
    """
    if ppmAxis is None:
        ppmAxis = np.linspace(0.5, 4.0, spec.shape[0])

    if ppmLim is not None:
        orig_ppm = ppmAxis                                    # save BEFORE first crop
        spec, ppmAxis = _crop_ppm(spec, orig_ppm, ppmLim)
        fit_ppm = orig_ppm if fit.shape[0] == orig_ppm.shape[0] \
                  else np.linspace(orig_ppm[0], orig_ppm[-1], fit.shape[0])
        fit, _ = _crop_ppm(fit, fit_ppm, ppmLim)
        if true is not None:
            true_ppm = orig_ppm if true.shape[0] == orig_ppm.shape[0] \
                       else np.linspace(orig_ppm[0], orig_ppm[-1], true.shape[0])
            true, _ = _crop_ppm(true, true_ppm, ppmLim)

    spec = _pick_part(spec, real)
    fit  = _pick_part(fit,  real)
    residual = spec - fit + 1.1 * spec.max()

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(ppmAxis, spec,     'k',       label='Data',     linewidth=1)
    ax.plot(ppmAxis, fit,      'r',       label='Fit',      alpha=0.6, linewidth=2)
    ax.plot(ppmAxis, residual, 'dimgray', label='Residual', alpha=0.8, linewidth=1)
    if true is not None:
        true = _pick_part(true, real)
        ax.plot(ppmAxis, true, 'g', label='True Spectrum',
                alpha=0.8, linewidth=1, linestyle='--')

    if not clean:
        ax.set_xlabel('Chemical Shift [ppm]')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style_mrs_ax(ax, clean)
    return _save_fig(fig, save_path, name)


def plot_spec(spec, ppmAxis=None, ppmLim=None, name='', save_path=None,
              real=True, color='k', title=None, clean=False):
    """Plot a single complex spectrum on a ppm axis (no fit / residual).

    Same visual style as plot_spec_and_fit - useful for sanity-checking
    the simulated input in isolation.

    Parameters
    ----------
    spec      : np.ndarray (complex)  spectrum to display (1-D).
    ppmAxis   : np.ndarray, optional  chemical-shift axis matching spec.
    ppmLim    : tuple(float, float), optional  (lo, hi) ppm window to display.
    name      : str  filename stem (without extension).
    save_path : str, optional  directory to save PNG into; created if missing.
    real      : bool  if True plot real part, else imaginary.
    color     : str  matplotlib color string for the spectrum line.
    title     : str, optional  axes title (small font, publication-friendly).
    clean     : bool  if True hide all axes/labels (publication snippet style).
    """
    if ppmAxis is None:
        ppmAxis = np.linspace(0.5, 4.0, spec.shape[0])

    if ppmLim is not None:
        spec, ppmAxis = _crop_ppm(spec, ppmAxis, ppmLim)

    spec = _pick_part(spec, real)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(ppmAxis, spec, color, label='Data', linewidth=1)
    if not clean:
        ax.set_xlabel('Chemical Shift [ppm]')
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style_mrs_ax(ax, clean)
    return _save_fig(fig, save_path, name)


# ══════════════════════════════════════════════════════════════════════════════
#  NIfTI-MRS batch plotting
# ══════════════════════════════════════════════════════════════════════════════


def vis_nifti_mrs_plus(
    data,
    display_dim=None,
    ppmlim=None,
    plot_avg=False,
    batch_index=None,
    max_batch_display=6,
    grid_layout=None,
    legend=True,
    figsize=None,
    title=None
):
    """
    Visualize NIfTI_MRS_Plus objects with batch-aware plotting.

    This function mirrors the NIfTI-MRS plot() method but adds batch support.

    Args:
        data: NIfTI_MRS_Plus object
        display_dim: Dimension to display (if multiple dims exist)
        ppmlim: Tuple of (min_ppm, max_ppm) for x-axis limits
        plot_avg: If True, plot average spectrum
        batch_index: Which batch element to plot (None = plot all up to max_batch_display)
        max_batch_display: Maximum number of batch elements to display
        grid_layout: Tuple (rows, cols) for grid layout. Auto if None
        legend: Whether to show legend
        figsize: Figure size (width, height)
        title: Plot title

    Returns:
        matplotlib figure object
    """
    from nifti_mrs_plus.core import NIfTI_MRS_Plus

    if not isinstance(data, NIfTI_MRS_Plus):
        raise TypeError("data must be a NIfTI_MRS_Plus object")

    # Get list of NIFTI_MRS objects
    nifti_list = data.list()

    if len(nifti_list) == 0:
        raise ValueError("NIfTI_MRS_Plus object is empty")

    # Determine ppm limits
    if ppmlim is None and len(nifti_list) > 0:
        first_nifti = nifti_list[0]
        try:
            nucleus = first_nifti.nucleus[0]
        except Exception:
            nucleus = "1H"
        ppmlim = _default_ppmlim(nucleus)

    # Handle batch indexing
    if batch_index is not None:
        # Plot single batch element
        if batch_index < 0 or batch_index >= len(nifti_list):
            raise IndexError(f"batch_index {batch_index} out of range [0, {len(nifti_list)})")

        nifti_to_plot = nifti_list[batch_index]

        return _plot_single_spectrum(
            nifti_to_plot,
            ppmlim=ppmlim,
            title=title or f"Batch Element {batch_index+1}/{len(nifti_list)}"
        )

    else:
        # Plot multiple batch elements
        n_to_plot = min(len(nifti_list), max_batch_display)

        if n_to_plot == 1:
            # Just plot the single element
            return vis_nifti_mrs_plus(
                data,
                display_dim=display_dim,
                ppmlim=ppmlim,
                plot_avg=plot_avg,
                batch_index=0,
                legend=legend,
                title=title
            )

        # Determine grid layout
        if grid_layout is None:
            if n_to_plot <= 2:
                rows, cols = 1, n_to_plot
            elif n_to_plot <= 4:
                rows, cols = 2, 2
            elif n_to_plot <= 6:
                rows, cols = 2, 3
            elif n_to_plot <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 3
        else:
            rows, cols = grid_layout

        # Create figure
        if figsize is None:
            figsize = (5 * cols, 3.5 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Plot each batch element
        for i in range(n_to_plot):
            nifti = nifti_list[i]
            ax = axes[i]

            # Plot spectrum on this axis
            _plot_spectrum_on_axis(
                nifti,
                ax=ax,
                ppmlim=ppmlim,
                title=f"Batch {i+1}",
                legend=False
            )

        # Hide unused subplots
        for i in range(n_to_plot, len(axes)):
            axes[i].axis('off')

        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        else:
            fig.suptitle(
                f"NIfTI_MRS_Plus Batch ({n_to_plot}/{len(nifti_list)} displayed)",
                fontsize=14,
                fontweight='bold',
                y=0.995
            )

        plt.tight_layout()
        return fig


def _plot_single_spectrum(nifti, ppmlim=(0.2, 4.2), title="MRS Spectrum"):
    """Plot a single NIfTI-MRS spectrum using basic matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    _plot_spectrum_on_axis(nifti, ax, ppmlim, title)
    plt.tight_layout()
    return fig


def _plot_spectrum_on_axis(nifti, ax, ppmlim=(0.2, 4.2), title="", legend=True):
    """Plot spectrum on axis, properly handling raw multi-coil/average data."""
    ppm, spec = _spectrum(nifti, ppmlim=ppmlim)

    # Plot
    ax.plot(ppm, np.real(spec), 'k-', linewidth=1.0, label='Real' if legend else None)

    # Formatting
    ax.invert_xaxis()
    if ppmlim is not None:
        ax.set_xlim(ppmlim[1], ppmlim[0])
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=10)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    if legend:
        ax.legend(loc='best', fontsize=8)


def plot_batch_comparison(
    nifti_plus,
    indices=None,
    ppmlim=(0.2, 4.2),
    labels=None,
    title="Batch Comparison",
    colors=None,
    alpha=0.7,
    figsize=(12, 5)
):
    """
    Plot multiple spectra from a batch overlaid for comparison.

    Args:
        nifti_plus: NIfTI_MRS_Plus object
        indices: List of batch indices to plot (None = all up to 6)
        ppmlim: PPM limits (min, max)
        labels: List of labels for each spectrum
        title: Plot title
        colors: List of colors for each spectrum
        alpha: Transparency for overlaid spectra
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    from nifti_mrs_plus.core import NIfTI_MRS_Plus

    if not isinstance(nifti_plus, NIfTI_MRS_Plus):
        raise TypeError("nifti_plus must be a NIfTI_MRS_Plus object")

    nifti_list = nifti_plus.list()

    if indices is None:
        indices = list(range(min(6, len(nifti_list))))

    if labels is None:
        labels = [f"Spectrum {i+1}" for i in indices]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    plotted = 0
    for j, idx in enumerate(indices):
        if idx >= len(nifti_list):
            warnings.warn(f"Index {idx} out of range, skipping")
            continue

        ppm, spec = _spectrum(nifti_list[idx], ppmlim=ppmlim)
        label = labels[j] if j < len(labels) else f"Spectrum {idx+1}"
        ax.plot(ppm, np.real(spec), color=colors[j], alpha=alpha,
                linewidth=1.0, label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError("No valid spectra to plot")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_xaxis()
    if ppmlim is not None:
        ax.set_xlim(max(ppmlim), min(ppmlim))
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=11)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    return fig


def plot_batch_grid_detailed(
    nifti_plus,
    max_display=9,
    ppmlim=(0.2, 4.2),
    title="Batch Grid",
    show_metabolites=False,
    figsize=None
):
    """
    Plot batch elements in a detailed grid with optional metabolite regions.

    Args:
        nifti_plus: NIfTI_MRS_Plus object
        max_display: Maximum number of spectra to display
        ppmlim: PPM limits
        title: Overall title
        show_metabolites: If True, highlight common metabolite regions
        figsize: Figure size (auto if None)

    Returns:
        matplotlib figure
    """
    from nifti_mrs_plus.core import NIfTI_MRS_Plus

    if not isinstance(nifti_plus, NIfTI_MRS_Plus):
        raise TypeError("nifti_plus must be a NIfTI_MRS_Plus object")

    nifti_list = nifti_plus.list()
    n_to_plot = min(len(nifti_list), max_display)

    # Grid layout
    if n_to_plot <= 3:
        rows, cols = 1, n_to_plot
    elif n_to_plot <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    if figsize is None:
        figsize = (5 * cols, 3.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Metabolite regions (1H MRS)
    metabolite_regions = {
        'NAA': (1.9, 2.1, '#E74C3C'),
        'Cr': (2.9, 3.1, '#3498DB'),
        'Cho': (3.1, 3.3, '#2ECC71'),
    }

    for i in range(n_to_plot):
        ax = axes[i]
        nifti = nifti_list[i]

        ppm, spec = _spectrum(nifti, ppmlim=ppmlim)

        # Plot
        ax.plot(ppm, np.real(spec), 'k-', linewidth=1.0)

        # Highlight metabolites
        if show_metabolites:
            for name, (ppm_min, ppm_max, color) in metabolite_regions.items():
                ax.axvspan(ppm_min, ppm_max, alpha=0.15, color=color)

        ax.invert_xaxis()
        if ppmlim:
            ax.set_xlim(ppmlim[1], ppmlim[0])
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(f"Sample {i+1}", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Only show x-label on bottom row
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Chemical Shift (ppm)", fontsize=9)

    # Hide unused subplots
    for i in range(n_to_plot, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    return fig


def quick_plot(nifti_plus, index=0, ppmlim=(0.2, 4.2)):
    """
    Quick plot of a single spectrum from batch.

    Args:
        nifti_plus: NIfTI_MRS_Plus object
        index: Batch index to plot
        ppmlim: PPM range

    Example:
        >>> quick_plot(nifti_plus, 0)  # Plot first spectrum
    """
    return vis_nifti_mrs_plus(nifti_plus, batch_index=index, ppmlim=ppmlim)
