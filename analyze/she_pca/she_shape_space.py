"""
Stages 1-2 of a 2D shape-analysis pipeline for yeast fluorescence videos:
per-frame Elliptic Fourier Descriptor (EFD, Kuhl & Giardina 1982) shape
encoding (Stage 1) and a per-frame, pooled-across-videos PCA shape space
(Stage 2). This is the 2D analog of the SHE -> PCA shape space in Viana
et al. 2023 (Allen Institute): EFD on a 2D contour plays the role SHE plays
on a 3D mesh, and per-frame PCA plays the role of their shape space.

Stage 3 (green/structure projection into the shape space, i.e. PILR) is
explicitly NOT implemented here.

This module is self-contained (does not import she_descriptor.py) so it can
run independently; the EFD math is the same published Kuhl-Giardina
formulas re-implemented here for this per-frame, pose-normalized use case.

Data: /datasets/yeast-imgs/generated_video/{gene_name}/{method}/{num_id}/{num_id}_video.gif
  (128x128 RGB, ~193 frames; red = nucleus reference marker, green = tagged
  structure; frame index is a shared biological clock across videos, so NO
  temporal alignment is performed here -- frame t means the same thing in
  every video). Note the on-disk filename is "{num_id}_video.gif", not
  "{num_id}.gif" as a naive path pattern would suggest.

Each gene has 3 generation methods (nucleus / random / structure) that are
different perturbation conditions, not repeated samples of the same
population -- pooling across them would mix distinct shape distributions
into one PCA. Every run is scoped to a single method (default: "nucleus").

Run: python she_shape_space.py [--method nucleus|random|structure]
     [--glob PATTERN] [--workers N] [--k K] ...
"""

import argparse
import functools
import glob
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import find_contours, label
from skimage.morphology import closing, opening, disk, dilation

OUT_DIR = Path(__file__).resolve().parent / "out"
GLOB_TEMPLATE = "/datasets/yeast-imgs/generated_video/YAL00*/{method}/*/*_video.gif"
DEFAULT_METHOD = "nucleus"

NUCLEUS_EFD_ORDER = 8
CELL_EFD_ORDER = 10
CELL_DIM = CELL_EFD_ORDER * 4    # 40
NUCLEUS_DIM = NUCLEUS_EFD_ORDER * 4  # 32

# Segmentation constants.
#
# Per-frame (unaveraged) Otsu thresholding is noisy enough that a real
# nucleus's raw binary mask often sheds a few tiny disconnected specks.
# Empirically (measured directly on this dataset) those speck components
# are 7-24 px, while every genuine single-nucleus component is >=189 px --
# a clean gap. NUCLEUS_MIN_OBJECT_AREA=5 (too small) let specks through and
# spuriously inflated n_nuclei (division false positives); 60 sits safely
# in the gap. The cell envelope mask did not show this problem (no
# multi-component cells were observed even at the original threshold), so
# CELL_MIN_OBJECT_AREA is left as a modest speck filter.
NUCLEUS_MIN_OBJECT_AREA = 60
CELL_MIN_OBJECT_AREA = 10
CLOSING_RADIUS = 2

# The final mask (post speck-removal + closing) was still visibly "peaky":
# closing only fills small concave gaps/holes, it does not trim small convex
# spikes sticking out of the blob -- those come from per-frame shot noise on
# the raw channel. A light pre-threshold Gaussian blur reduces that noise at
# the source, and a small morphological opening after cleanup trims whatever
# spikes remain. Both are kept small so the real bud neck (a genuine, larger-
# scale concave feature) survives.
#
# sigma=1.0 measurably inflates area (e.g. 179->214 px, ~20%, on one tested
# nucleus) because blurring spreads a compact bright core outward before any
# threshold is applied -- and since it's a fixed *absolute* blur radius, it
# inflates the smaller object (nucleus, r~7-10px) proportionally more than
# the larger one (cell, r~20-25px), directly biasing the nucleus/cell size
# ratio upward. 0.5 keeps most of the peak-suppression benefit (checked
# against segmentation_examples.png) while roughly halving that inflation.
SEGMENT_BLUR_SIGMA = 0.5
OPENING_RADIUS = 1

# A single global Otsu threshold on the nucleus channel is unstable: on some
# frames a dim, diffuse halo around the bright compact nucleus core gets
# included, roughly doubling the segmented area (measured directly: 999 px
# vs 266 px for two otherwise-comparable nuclei -- a real, large effect, not
# noise). The otsu-threshold/max-intensity ratio varies continuously across
# frames (5th-95th percentile ~0.16-0.32), so a fixed floor would just as
# often clip legitimately-segmented frames; the actual fix is a second Otsu
# pass restricted to a dilated ROI around the first-pass mask, which forces
# Otsu to separate "core vs halo" using only nearby pixel values instead of
# "anything vs the true (near-zero) background" -- the latter is too easy a
# split and lets the halo through as "foreground".
NUCLEUS_ROI_DILATION_RADIUS = 6

# A budding yeast cell has at most one mitotic pair of nuclei -- 3+
# components are a segmentation artifact, not real biology. Checked
# directly on several such frames: there are always two genuine
# nucleus-scale components (110-332 px) plus one small extra (12-60 px,
# a thin connecting-bridge fragment or debris), never three comparable
# ones. Cap to the largest 2 rather than inventing a size-based debris
# filter here, since remove_small_components/opening already do that at
# the pixel level and clearly aren't enough for this particular artifact.
MAX_NUCLEI_COMPONENTS = 2

PCA_K = 8


# ----------------------------------------------------------------------------
# Path parsing + I/O
# ----------------------------------------------------------------------------


def parse_gif_path(gif_path):
    p = Path(gif_path)
    num_id = p.parent.name
    method = p.parent.parent.name
    gene_name = p.parent.parent.parent.name
    return gene_name, method, num_id


def load_gif_channels(gif_path):
    """Load a GIF and return all frames as an (T, H, W, 3) uint8 array."""
    im = Image.open(gif_path)
    frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    return np.stack(frames, axis=0)


def load_single_frame(gif_path, frame_idx):
    im = Image.open(gif_path)
    im.seek(frame_idx)
    return np.array(im.convert("RGB"))


# ----------------------------------------------------------------------------
# Stage 1 step 2-4: segmentation + contour extraction
# ----------------------------------------------------------------------------


def remove_small_components(binary, min_area):
    """Drop connected components with area <= min_area.

    Self-contained (label + bincount) rather than skimage's
    remove_small_objects, whose size kwarg was renamed min_size -> max_size
    between skimage versions (a TypeError on one or the other depending on
    which is installed); this works identically across versions.
    """
    labeled = label(binary)
    if labeled.max() == 0:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts > min_area
    return keep[labeled]


def keep_largest_k_components(binary, k):
    """Keep only the k largest connected components, dropping the rest."""
    labeled = label(binary)
    if labeled.max() <= k:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep_labels = np.argsort(-counts)[:k]
    return np.isin(labeled, keep_labels)


def segment_nucleus(red):
    """Blur -> Otsu -> remove tiny objects -> second, ROI-local Otsu pass
    (to strip any diffuse halo the first pass let through) -> open -> close.

    Returns (mask, n_nuclei). n_nuclei is the post-cleanup component count
    and doubles as the division detector (>1 nucleus => division/mitosis).
    """
    if red.max() <= red.min():
        return np.zeros_like(red, dtype=bool), 0
    red_smooth = gaussian(red, sigma=SEGMENT_BLUR_SIGMA, preserve_range=True)

    rough_thresh = threshold_otsu(red_smooth)
    rough_mask = red_smooth > rough_thresh
    rough_mask = remove_small_components(rough_mask, NUCLEUS_MIN_OBJECT_AREA)
    if not rough_mask.any():
        return np.zeros_like(red, dtype=bool), 0

    roi = dilation(rough_mask, footprint=disk(NUCLEUS_ROI_DILATION_RADIUS))
    roi_vals = red_smooth[roi]
    fine_thresh = threshold_otsu(roi_vals) if roi_vals.max() > roi_vals.min() else rough_thresh
    binary = (red_smooth > fine_thresh) & roi

    binary = remove_small_components(binary, NUCLEUS_MIN_OBJECT_AREA)
    binary = opening(binary, footprint=disk(OPENING_RADIUS))
    closed = closing(binary, footprint=disk(CLOSING_RADIUS))
    closed = keep_largest_k_components(closed, MAX_NUCLEI_COMPONENTS)
    labeled = label(closed)
    n_nuclei = int(labeled.max())
    return closed, n_nuclei


def segment_cell(red, green):
    """Blur -> threshold (red+green) envelope > 5 -> keep largest-ish
    components -> open -> close. Returns (mask, n_cell_components).

    "Largest-ish" keeps every component above a small area floor rather than
    only the single largest, because a genuinely dividing cell can show two
    substantial, separate cell-sized blobs (mother + daughter) -- collapsing
    to one component would hide exactly the division events we want to see.
    """
    envelope = red + green
    envelope_smooth = gaussian(envelope, sigma=SEGMENT_BLUR_SIGMA, preserve_range=True)
    binary = envelope_smooth > 5
    binary = remove_small_components(binary, CELL_MIN_OBJECT_AREA)
    binary = opening(binary, footprint=disk(OPENING_RADIUS))
    closed = closing(binary, footprint=disk(CLOSING_RADIUS))
    labeled = label(closed)
    n_cell_components = int(labeled.max())
    return closed, n_cell_components


def single_component_contour(mask):
    """Extract the contour of a mask expected to hold exactly one component
    (caller has already verified this). Returns an ordered (K, 2) (x, y)
    array, or None if no contour could be traced.
    """
    contours_rc = find_contours(mask.astype(float), level=0.5)
    if not contours_rc:
        return None
    longest = max(contours_rc, key=len)
    contour_xy = longest[:, ::-1].copy()
    if np.allclose(contour_xy[0], contour_xy[-1]):
        contour_xy = contour_xy[:-1]
    return contour_xy


def polygon_area(contour_xy):
    x, y = contour_xy[:, 0], contour_xy[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(contour_xy):
    """Area-weighted (shoelace) centroid -- the correct geometric center of
    the enclosed shape, not just the mean of the (non-uniformly spaced)
    contour vertices.
    """
    x, y = contour_xy[:, 0], contour_xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a2 = cross.sum()
    if abs(a2) < 1e-9:
        return contour_xy.mean(axis=0)
    cx = np.sum((x + x1) * cross) / (3 * a2)
    cy = np.sum((y + y1) * cross) / (3 * a2)
    return np.array([cx, cy])


def equivalent_radius(area):
    return np.sqrt(max(area, 0.0) / np.pi)


def labeled_component_contours(mask):
    """For a mask that may hold multiple components (e.g. two post-mitosis
    nuclei), return one (contour_xy, area) pair per component, largest
    first. Each contour is traced on that component's own isolated
    sub-mask so neighboring components can't leak into it.
    """
    labeled = label(mask)
    results = []
    for lbl in range(1, labeled.max() + 1):
        contour = single_component_contour(labeled == lbl)
        if contour is None or len(contour) < 10:
            continue
        results.append((contour, polygon_area(contour)))
    results.sort(key=lambda r: -r[1])
    return results


def validity_reason(n_nuclei, n_cell_components):
    """Empty string = valid_single. Requires exactly one cell component --
    across every frame processed so far (tens of thousands), the cell
    envelope was NEVER observed to split into >1 component even during
    division (mother/daughter stay bridged by a neck until cytokinesis,
    which lags nuclear division), so a dividing cell shows up as one cell
    body containing two nuclei, not two cells. The nucleus may therefore
    be 1 (interphase) or 2 (post-mitosis, pre-cytokinesis); 0 or >2 is
    still invalid (no correspondence model for more than 2).
    """
    if n_cell_components == 0:
        return "no cell region detected"
    if n_cell_components > 1:
        return f"n_cell_components={n_cell_components} (division/multi-cell)"
    if n_nuclei == 0:
        return "no nucleus detected"
    if n_nuclei > 2:
        return f"n_nuclei={n_nuclei} (more than 2 nuclei, unsupported)"
    return ""


# ----------------------------------------------------------------------------
# Stage 1 step 5: pose normalization (translate + rotate; scale kept)
# ----------------------------------------------------------------------------


def normalize_pose(contour_xy, normalize_rotation=True, normalize_scale=False):
    """Translate to centroid; optionally rotate so the major axis -> x.

    Scale is kept by default (size is a meaningful shape-space mode, as in
    the 3D paper) but the RMS radius is still computed and returned as pose
    metadata so a caller can rescale later if they want a scale-normalized
    variant. Both normalizations are toggleable.

    A line (the major axis, from PCA/eigen-decomposition of the point
    covariance) has no inherent direction, so aligning it to +x leaves a
    180-degree ambiguity; we resolve it deterministically by requiring the
    farthest contour point from the centroid to land on the +x side. This
    keeps the pose-normalized frame stable frame-to-frame (no sign is
    otherwise fixed by eigh's returned eigenvector). We do NOT additionally
    resolve the up/down (y-mirror) ambiguity, since doing so would erase
    genuine left/right shape asymmetry (e.g. off-axis bud position).

    Returns (contour_normalized, pose) where pose = {centroid, angle, scale}
    and `angle` is the actual total rotation applied (0 if
    normalize_rotation is False, but still computed for reference).
    """
    centroid = polygon_centroid(contour_xy)
    centered = contour_xy - centroid
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.arctan2(major[1], major[0]))

    out = centered
    applied_angle = 0.0
    if normalize_rotation:
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        rotated = centered @ rot.T
        # resolve the 180-degree axis-direction ambiguity
        far_idx = np.argmax(np.sum(rotated ** 2, axis=1))
        if rotated[far_idx, 0] < 0:
            rotated = -rotated
            applied_angle = angle + np.pi
        else:
            applied_angle = angle
        out = rotated

    if normalize_scale and scale > 0:
        out = out / scale

    pose = {"centroid": centroid, "angle": applied_angle, "scale": scale}
    return out, pose


def denormalize_pose(contour_norm, pose, scale_was_applied):
    """Invert normalize_pose's translation/rotation/scale for QC plotting."""
    out = contour_norm.copy()
    if scale_was_applied:
        out = out * pose["scale"]
    c, s = np.cos(pose["angle"]), np.sin(pose["angle"])
    rot = np.array([[c, -s], [s, c]])
    out = out @ rot.T
    out = out + pose["centroid"]
    return out


# ----------------------------------------------------------------------------
# Stage 1 step 6: EFD encode/decode (Kuhl & Giardina, 1982)
# ----------------------------------------------------------------------------
# The per-object shape descriptor is the (order, 4) [a_n, b_n, c_n, d_n]
# harmonic coefficients only -- the DC/centroid term is pose, not shape, and
# is stored separately (see normalize_pose above), matching the paper's
# separation of "where/how big" from "what shape".


def efd_coeffs(contour_xy, order):
    contour_closed = np.vstack([contour_xy, contour_xy[0]])
    dxy = np.diff(contour_closed, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt = np.where(dt == 0, 1e-10, dt)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]

    n = np.arange(1, order + 1).reshape(-1, 1)
    phi = (2.0 * np.pi * n * t.reshape(1, -1)) / T
    d_cos = np.cos(phi[:, 1:]) - np.cos(phi[:, :-1])
    d_sin = np.sin(phi[:, 1:]) - np.sin(phi[:, :-1])
    const = T / (2.0 * (n[:, 0] ** 2) * np.pi ** 2)

    a = const * np.sum((dxy[:, 0] / dt) * d_cos, axis=1)
    b = const * np.sum((dxy[:, 0] / dt) * d_sin, axis=1)
    c = const * np.sum((dxy[:, 1] / dt) * d_cos, axis=1)
    d = const * np.sum((dxy[:, 1] / dt) * d_sin, axis=1)
    return np.stack([a, b, c, d], axis=1)


def efd_phase_normalize(coeffs):
    """Remove the arbitrary parametrization starting-point phase (the other
    half of Kuhl & Giardina's normalization; normalize_pose already handles
    the spatial rotation/translation half).

    efd_coeffs's t=0 is wherever find_contours happened to start tracing --
    an accident of the raw mask's pixel layout, uncorrelated across
    different cells/frames. A shift of the starting point by fraction delta
    of the perimeter rotates harmonic n's (a_n, b_n) and (c_n, d_n) pairs by
    angle n*delta*2*pi, changing nothing about the physical shape traced,
    only which point is labeled t=0. Left uncorrected, coefficients from
    different individuals point in effectively random directions in
    coefficient space, so their arithmetic mean (as used for the pooled
    Stage 2 PCA/mean shape) partially cancels instead of averaging -- this
    is exactly what was collapsing the pooled mean nucleus at some frames.

    Fixes the phase via the first harmonic's ellipse (the standard
    Kuhl-Giardina reference direction), which has a 2-fold ambiguity (theta
    and theta+pi both solve the defining equation, since it comes from a
    doubled-angle formula); resolved deterministically by requiring a_1>=0.

    Returns (coeffs_normalized, delta) where delta is the applied starting-
    point shift as a fraction of the full loop (in [0, 1)) -- callers that
    need point-for-point correspondence with the original contour (e.g. QC
    RMS error) must shift their t values by -delta to compensate.
    """
    a1, b1, c1, d1 = coeffs[0]
    theta1 = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)

    def apply(theta):
        n = np.arange(1, coeffs.shape[0] + 1)
        cos_n, sin_n = np.cos(n * theta), np.sin(n * theta)
        a, b, c, d = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
        a_new = a * cos_n + b * sin_n
        b_new = -a * sin_n + b * cos_n
        c_new = c * cos_n + d * sin_n
        d_new = -c * sin_n + d * cos_n
        return np.stack([a_new, b_new, c_new, d_new], axis=1)

    out = apply(theta1)
    if out[0, 0] < 0:
        theta1 = theta1 + np.pi
        out = apply(theta1)
    delta = (theta1 / (2.0 * np.pi)) % 1.0
    return out, delta


def efd_encode(contour_xy, order):
    """Encode + phase-normalize: the Stage-1-facing entry point that both
    describe_frame and the QC code should use so every stored coefficient
    vector is in the same phase convention (see efd_phase_normalize).

    Returns (coeffs, delta) -- delta is 0 only in the degenerate case where
    the shape has no meaningful first harmonic; callers doing point-matched
    comparisons against the original contour must account for it.
    """
    coeffs = efd_coeffs(contour_xy, order)
    return efd_phase_normalize(coeffs)


def efd_reconstruct(coeffs, dc, t_norm, order=None):
    if order is None:
        order = coeffs.shape[0]
    order = min(order, coeffs.shape[0])
    t_norm = np.asarray(t_norm)
    x = np.full_like(t_norm, dc[0], dtype=float)
    y = np.full_like(t_norm, dc[1], dtype=float)
    for i in range(order):
        n = i + 1
        a, b, c, d = coeffs[i]
        ang = 2.0 * np.pi * n * t_norm
        x += a * np.cos(ang) + b * np.sin(ang)
        y += c * np.cos(ang) + d * np.sin(ang)
    return np.stack([x, y], axis=1)


def efd_decode(coeffs, dc=(0.0, 0.0), num_points=200, order=None):
    """Dense closed-curve reconstruction for plotting/mean-shape use."""
    t_dense = np.linspace(0, 1, num_points, endpoint=False)
    return efd_reconstruct(coeffs, dc, t_dense, order=order)


def contour_arc_length_params(contour_xy):
    contour_closed = np.vstack([contour_xy, contour_xy[0]])
    dxy = np.diff(contour_closed, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt = np.where(dt == 0, 1e-10, dt)
    t = np.concatenate([[0.0], np.cumsum(dt)])[:-1]
    return t / t[-1] if t[-1] > 0 else t


def rms_reconstruction_error(coeffs, contour_norm, order, phase_delta=0.0):
    """RMS error (px) between a pose-normalized contour and its EFD
    reconstruction at the same arc-length parameters. Since normalize_pose
    only translates/rotates (an isometry) and by default does not rescale,
    this RMS equals the RMS error in original image pixels.

    `phase_delta` is the starting-point shift efd_phase_normalize applied
    (if `coeffs` came from efd_encode) -- the reconstruction must be sampled
    at (t - phase_delta) to land on the same physical points as
    contour_norm's own (unshifted) vertex parametrization.
    """
    t_norm = contour_arc_length_params(contour_norm)
    recon = efd_reconstruct(coeffs, (0.0, 0.0), (t_norm - phase_delta) % 1.0, order=order)
    err = np.sqrt(((recon - contour_norm) ** 2).sum(axis=1))
    return float(np.sqrt(np.mean(err ** 2)))


# ----------------------------------------------------------------------------
# Stage 1: per-frame + per-video processing
# ----------------------------------------------------------------------------

RECORD_FIELDS = [
    "n_nuclei", "n_cell_components", "valid_single", "reason",
    "cell_efd", "nucleus_efd", "nucleus_efd_2",
    "cell_centroid", "cell_angle", "cell_scale",
    "nucleus_centroid", "nucleus_angle", "nucleus_scale",
    "nucleus_centroid_2", "nucleus_angle_2", "nucleus_scale_2",
    "cell_area", "nucleus_area", "nucleus_area_2",
    "cell_r_eq", "nucleus_r_eq", "nucleus_r_eq_2",
    "cell_contour_points", "nucleus_contour_points", "nucleus_contour_points_2",
]


def _empty_record(reason):
    return {
        "n_nuclei": 0, "n_cell_components": 0, "valid_single": False, "reason": reason,
        "cell_efd": np.full((CELL_EFD_ORDER, 4), np.nan),
        "nucleus_efd": np.full((NUCLEUS_EFD_ORDER, 4), np.nan),
        "nucleus_efd_2": np.full((NUCLEUS_EFD_ORDER, 4), np.nan),
        "cell_centroid": np.full(2, np.nan), "cell_angle": np.nan, "cell_scale": np.nan,
        "nucleus_centroid": np.full(2, np.nan), "nucleus_angle": np.nan, "nucleus_scale": np.nan,
        "nucleus_centroid_2": np.full(2, np.nan), "nucleus_angle_2": np.nan, "nucleus_scale_2": np.nan,
        "cell_area": np.nan, "nucleus_area": np.nan, "nucleus_area_2": np.nan,
        "cell_r_eq": np.nan, "nucleus_r_eq": np.nan, "nucleus_r_eq_2": np.nan,
        "cell_contour_points": 0, "nucleus_contour_points": 0, "nucleus_contour_points_2": 0,
    }


def describe_frame(rgb, normalize_rotation=True, normalize_scale=False):
    """Segment + pose-normalize + EFD-encode one (H, W, 3) frame.

    Returns a dict with the fields in RECORD_FIELDS. Fields are NaN/0 when
    the frame isn't valid_single -- n_nuclei, n_cell_components,
    valid_single and reason are still always populated so multi-component
    frames are never silently dropped.

    "nucleus_*" describes the larger nucleus (or the only one, in
    interphase); "nucleus_*_2" describes the smaller one, populated only
    when n_nuclei==2 (post-mitosis, pre-cytokinesis -- see validity_reason
    for why that's one cell with two nuclei here, not two cells).
    """
    red = rgb[..., 0].astype(np.float64)
    green = rgb[..., 1].astype(np.float64)

    nucleus_mask, n_nuclei = segment_nucleus(red)
    cell_mask, n_cell_components = segment_cell(red, green)
    reason = validity_reason(n_nuclei, n_cell_components)
    valid_single = reason == ""

    record = _empty_record(reason)
    record["n_nuclei"] = n_nuclei
    record["n_cell_components"] = n_cell_components

    if not valid_single:
        return record

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nucleus_components = labeled_component_contours(nucleus_mask)
        cell_contour = single_component_contour(cell_mask)

    if (len(nucleus_components) < n_nuclei or cell_contour is None
            or len(cell_contour) < 10):
        record["reason"] = "contour extraction failed"
        record["valid_single"] = False
        return record

    cell_area = polygon_area(cell_contour)
    cell_norm, cell_pose = normalize_pose(cell_contour, normalize_rotation, normalize_scale)
    cell_efd, _ = efd_encode(cell_norm, order=CELL_EFD_ORDER)
    record.update({
        "cell_efd": cell_efd,
        "cell_centroid": cell_pose["centroid"], "cell_angle": cell_pose["angle"], "cell_scale": cell_pose["scale"],
        "cell_area": cell_area, "cell_r_eq": equivalent_radius(cell_area),
        "cell_contour_points": len(cell_contour),
    })

    for (nucleus_contour, nucleus_area), suffix in zip(nucleus_components, ("", "_2")):
        nucleus_norm, nucleus_pose = normalize_pose(nucleus_contour, normalize_rotation, normalize_scale)
        nucleus_efd, _ = efd_encode(nucleus_norm, order=NUCLEUS_EFD_ORDER)
        record.update({
            f"nucleus_efd{suffix}": nucleus_efd,
            f"nucleus_centroid{suffix}": nucleus_pose["centroid"],
            f"nucleus_angle{suffix}": nucleus_pose["angle"],
            f"nucleus_scale{suffix}": nucleus_pose["scale"],
            f"nucleus_area{suffix}": nucleus_area,
            f"nucleus_r_eq{suffix}": equivalent_radius(nucleus_area),
            f"nucleus_contour_points{suffix}": len(nucleus_contour),
        })

    record["valid_single"] = True
    return record


def process_video(gif_path, normalize_rotation=True, normalize_scale=False):
    """Process every frame of one video. Never raises: any per-frame
    failure is caught and recorded as an invalid frame with a reason.
    """
    gene_name, method, num_id = parse_gif_path(gif_path)
    video_id = f"{gene_name}_{method}_{num_id}"

    try:
        frames_rgb = load_gif_channels(gif_path)
    except Exception as exc:  # noqa: BLE001 - must never crash the batch
        print(f"WARNING: failed to load {gif_path}: {exc}")
        return []

    records = []
    for t in range(frames_rgb.shape[0]):
        try:
            rec = describe_frame(frames_rgb[t], normalize_rotation, normalize_scale)
        except Exception as exc:  # noqa: BLE001
            rec = _empty_record(f"exception: {exc}")
        rec["gene_name"] = gene_name
        rec["method"] = method
        rec["num_id"] = num_id
        rec["video_id"] = video_id
        rec["gif_path"] = str(gif_path)
        rec["frame_idx"] = t
        records.append(rec)
    return records


# ----------------------------------------------------------------------------
# Stage 1: table assembly, QC, saving
# ----------------------------------------------------------------------------


def records_to_table(records):
    keys = RECORD_FIELDS + ["gene_name", "method", "num_id", "video_id", "gif_path", "frame_idx"]
    table = {}
    for key in keys:
        values = [r[key] for r in records]
        table[key] = np.array(values)
    return table


def run_stage1(pattern, out_dir, workers, normalize_rotation, normalize_scale, qc_samples, seed):
    gif_paths = sorted(glob.glob(pattern))
    print(f"Stage 1: found {len(gif_paths)} videos matching '{pattern}'")
    if not gif_paths:
        raise SystemExit("No videos matched the glob pattern; nothing to do.")

    worker = functools.partial(
        process_video, normalize_rotation=normalize_rotation, normalize_scale=normalize_scale
    )

    all_records = []
    if workers <= 1:
        for i, p in enumerate(gif_paths):
            all_records.extend(worker(p))
            if (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(gif_paths)} videos")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, recs in enumerate(ex.map(worker, gif_paths)):
                all_records.extend(recs)
                if (i + 1) % 25 == 0:
                    print(f"  processed {i + 1}/{len(gif_paths)} videos")

    table = records_to_table(all_records)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "descriptors.npz", **table)

    _print_stage1_summary(table, gif_paths)
    _plot_segmentation_examples(table, out_dir, qc_samples, seed)
    _plot_stage1_qc(table, out_dir, qc_samples, seed, normalize_rotation, normalize_scale)

    return table


def _plot_segmentation_examples(table, out_dir, n_samples, seed):
    """Show the raw segmentation itself (Otsu binaries + final masks), not
    just the downstream EFD fit -- a good-looking EFD overlay can hide a bad
    mask if the fit is judged only against its own segmented contour. Mixes
    in a few flagged (division/multi-component) frames for contrast.
    """
    rng = np.random.default_rng(seed)
    n_flagged = min(n_samples // 2, int((~table["valid_single"]).sum()))
    n_valid = n_samples - n_flagged

    valid_idx = np.flatnonzero(table["valid_single"])
    flagged_idx = np.flatnonzero(~table["valid_single"])
    picks = []
    if n_valid > 0 and len(valid_idx) > 0:
        picks.extend(rng.choice(valid_idx, size=min(n_valid, len(valid_idx)), replace=False))
    if n_flagged > 0 and len(flagged_idx) > 0:
        picks.extend(rng.choice(flagged_idx, size=n_flagged, replace=False))
    if not picks:
        print("Segmentation QC: no frames available to sample; skipping.")
        return

    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 3.1 * n_rows), squeeze=False)
    col_titles = ["RGB frame", "red + Otsu (raw)", "nucleus mask (final)",
                  "red+green + thresh>5 (raw)", "cell mask (final)"]

    for i, idx in enumerate(picks):
        gif_path = table["gif_path"][idx]
        frame_idx = int(table["frame_idx"][idx])
        rgb = load_single_frame(gif_path, frame_idx)
        red = rgb[..., 0].astype(np.float64)
        green = rgb[..., 1].astype(np.float64)
        envelope = red + green

        nucleus_raw = red > threshold_otsu(red) if red.max() > red.min() else np.zeros_like(red, bool)
        nucleus_final, n_nuc = segment_nucleus(red)
        cell_raw = envelope > 5
        cell_final, n_cell = segment_cell(red, green)

        row_label = (f"{table['video_id'][idx]}  frame {frame_idx}\n"
                     f"n_nuclei={n_nuc}, n_cell_components={n_cell}"
                     + (f"\n[{table['reason'][idx]}]" if table["reason"][idx] else ""))

        ax = axes[i, 0]
        ax.imshow(rgb)
        ax.set_ylabel(row_label, fontsize=7, rotation=0, ha="right", va="center", labelpad=60)

        ax = axes[i, 1]
        ax.imshow(red, cmap="gray")
        ax.contour(nucleus_raw, levels=[0.5], colors="yellow", linewidths=1)

        ax = axes[i, 2]
        ax.imshow(red, cmap="gray")
        ax.contour(nucleus_final, levels=[0.5], colors="cyan", linewidths=1.2)

        ax = axes[i, 3]
        ax.imshow(envelope, cmap="gray")
        ax.contour(cell_raw, levels=[0.5], colors="yellow", linewidths=1)

        ax = axes[i, 4]
        ax.imshow(envelope, cmap="gray")
        ax.contour(cell_final, levels=[0.5], colors="magenta", linewidths=1.2)

        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=9)

    fig.tight_layout()
    path = Path(out_dir) / "segmentation_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSegmentation QC: saved {n_rows} example frames ({n_flagged} flagged, "
          f"{n_rows - n_flagged} valid_single) to {path}")


def _print_stage1_summary(table, gif_paths):
    total = len(table["frame_idx"])
    valid = int(table["valid_single"].sum())
    division = int((table["n_nuclei"] == 2).sum())
    print(f"\nStage 1 summary:")
    print(f"  videos: {len(gif_paths)}, total frames: {total}")
    print(f"  valid_single frames: {valid} ({100 * valid / total:.1f}%)")
    print(f"  frames with n_nuclei==2 (division): {division} ({100 * division / total:.1f}%)")

    video_ids = table["video_id"]
    unique_videos = np.unique(video_ids)
    fracs = []
    for vid in unique_videos:
        m = video_ids == vid
        fracs.append(table["valid_single"][m].mean())
    fracs = np.array(fracs)
    print(f"  per-video valid fraction: min={fracs.min():.2f}, "
          f"mean={fracs.mean():.2f}, max={fracs.max():.2f}")
    low = unique_videos[fracs < 0.3]
    if len(low):
        print(f"  {len(low)} videos with <30% valid frames (heavy division/segmentation "
              f"trouble), e.g.: {list(low[:5])}")

    if valid / total < 0.5:
        reasons, reason_counts = np.unique(table["reason"][~table["valid_single"]], return_counts=True)
        order = np.argsort(-reason_counts)
        print("  WARNING: less than half of all frames are valid -- top failure reasons "
              "(likely an environment/library issue, not real biology):")
        for r, c in zip(reasons[order][:5], reason_counts[order][:5]):
            print(f"    {c:>6}x  {r}")


def _plot_stage1_qc(table, out_dir, n_samples, seed, normalize_rotation, normalize_scale):
    rng = np.random.default_rng(seed)
    # Restricted to n_nuclei==1: this plot's reconstruction machinery
    # (single_component_contour) traces just one nucleus contour, so
    # sampling an n_nuclei==2 (valid_single) frame would silently show
    # only the larger of the two nuclei with no indication of the other.
    valid_idx = np.flatnonzero(table["valid_single"] & (table["n_nuclei"] == 1))
    if len(valid_idx) == 0:
        print("Stage 1 QC: no valid_single frames available to sample; skipping QC plot.")
        return
    n_samples = min(n_samples, len(valid_idx))
    sample_idx = rng.choice(valid_idx, size=n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(3.2 * n_samples, 6.4), squeeze=False)
    nucleus_rms, cell_rms = [], []

    for j, idx in enumerate(sample_idx):
        gif_path = table["gif_path"][idx]
        frame_idx = int(table["frame_idx"][idx])
        rgb = load_single_frame(gif_path, frame_idx)
        red = rgb[..., 0].astype(np.float64)
        green = rgb[..., 1].astype(np.float64)

        nucleus_mask, _ = segment_nucleus(red)
        cell_mask, _ = segment_cell(red, green)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nucleus_contour = single_component_contour(nucleus_mask)
            cell_contour = single_component_contour(cell_mask)

        nucleus_norm, nucleus_pose = normalize_pose(nucleus_contour, normalize_rotation, normalize_scale)
        cell_norm, cell_pose = normalize_pose(cell_contour, normalize_rotation, normalize_scale)
        nucleus_efd, nucleus_delta = efd_encode(nucleus_norm, order=NUCLEUS_EFD_ORDER)
        cell_efd, cell_delta = efd_encode(cell_norm, order=CELL_EFD_ORDER)

        n_rms = rms_reconstruction_error(nucleus_efd, nucleus_norm, NUCLEUS_EFD_ORDER, phase_delta=nucleus_delta)
        c_rms = rms_reconstruction_error(cell_efd, cell_norm, CELL_EFD_ORDER, phase_delta=cell_delta)
        nucleus_rms.append(100 * n_rms / max(table["nucleus_r_eq"][idx], 1e-9))
        cell_rms.append(100 * c_rms / max(table["cell_r_eq"][idx], 1e-9))

        nucleus_recon_norm = efd_decode(nucleus_efd, order=NUCLEUS_EFD_ORDER)
        cell_recon_norm = efd_decode(cell_efd, order=CELL_EFD_ORDER)
        nucleus_recon_img = denormalize_pose(nucleus_recon_norm, nucleus_pose, normalize_scale)
        cell_recon_img = denormalize_pose(cell_recon_norm, cell_pose, normalize_scale)

        background = np.stack([
            np.clip(red / max(red.max(), 1e-6), 0, 1),
            np.clip(green / max(green.max(), 1e-6), 0, 1),
            np.zeros_like(red),
        ], axis=-1)

        title = f"{table['video_id'][idx]}\nframe {frame_idx}"

        ax = axes[0, j]
        ax.imshow(background)
        ax.plot(nucleus_contour[:, 0], nucleus_contour[:, 1], "w-", lw=1.2, label="true")
        ax.plot(nucleus_recon_img[:, 0], nucleus_recon_img[:, 1], "c--", lw=1.4, label="EFD N=8")
        ax.set_title(f"nucleus\n{title}", fontsize=8)
        ax.legend(loc="upper right", fontsize=5)
        ax.axis("off")

        ax = axes[1, j]
        ax.imshow(background)
        ax.plot(cell_contour[:, 0], cell_contour[:, 1], "w-", lw=1.2, label="true")
        ax.plot(cell_recon_img[:, 0], cell_recon_img[:, 1], "m--", lw=1.4, label="EFD N=10")
        ax.set_title("cell", fontsize=8)
        ax.legend(loc="upper right", fontsize=5)
        ax.axis("off")

    fig.tight_layout()
    qc_path = Path(out_dir) / "efd_recon_qc.png"
    fig.savefig(qc_path, dpi=150)
    plt.close(fig)

    print(f"\nStage 1 QC ({n_samples} sampled valid_single frames):")
    print(f"  nucleus RMS error: mean={np.mean(nucleus_rms):.2f}% of r_eq "
          f"(target <~2% at N={NUCLEUS_EFD_ORDER})")
    print(f"  cell RMS error:    mean={np.mean(cell_rms):.2f}% of r_eq "
          f"(target <~2% at N={CELL_EFD_ORDER})")
    print(f"  saved: {qc_path}")


# ----------------------------------------------------------------------------
# Stage 2: per-frame pooled PCA shape space
# ----------------------------------------------------------------------------


def build_shape_space(table, frame_idx, k=PCA_K):
    """Fit one PCA on the pooled [cell_efd, nucleus_efd] descriptors of all
    valid_single videos at a single frame index. gene_name is metadata only
    and never enters the fit -- one pooled coordinate frame per frame index,
    shared across every label.
    """
    mask = (table["frame_idx"] == frame_idx) & table["valid_single"]
    n_valid = int(mask.sum())
    video_ids = table["video_id"][mask]

    result = {
        "frame": frame_idx, "n_valid": n_valid, "video_ids": video_ids,
        "mean": None, "components": None, "explained_variance_ratio": None,
        "scores": None, "k_used": 0,
    }
    if n_valid < 3:
        return result

    cell_efd = table["cell_efd"][mask].reshape(n_valid, -1)
    nucleus_efd = table["nucleus_efd"][mask].reshape(n_valid, -1)
    x = np.concatenate([cell_efd, nucleus_efd], axis=1)

    k_use = min(k, n_valid - 1, x.shape[1])
    if k_use < 1:
        return result

    pca = PCA(n_components=k_use, random_state=0)
    scores = pca.fit_transform(x)
    components = pca.components_.copy()

    # Deterministic sign fix: larger cell -> positive score, so the same PC
    # axis means the same thing across frames/runs instead of an arbitrary
    # sign flip.
    cell_r_eq = table["cell_r_eq"][mask]
    for j in range(k_use):
        if np.std(scores[:, j]) > 0 and np.std(cell_r_eq) > 0:
            corr = np.corrcoef(scores[:, j], cell_r_eq)[0, 1]
            if not np.isnan(corr) and corr < 0:
                scores[:, j] *= -1
                components[j] *= -1

    result.update({
        "mean": pca.mean_, "components": components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "scores": scores, "k_used": k_use,
    })
    return result


def mean_shape_contours(pca_result):
    """Inverse-EFD of the pooled mean vector -> (cell_contour, nucleus_contour).
    Both are in the pose-normalized (centered, rotation-aligned) frame since
    that's the coordinate system the PCA operates in.
    """
    if pca_result["mean"] is None:
        return None, None
    mean = pca_result["mean"]
    cell_coeffs = mean[:CELL_DIM].reshape(CELL_EFD_ORDER, 4)
    nucleus_coeffs = mean[CELL_DIM:CELL_DIM + NUCLEUS_DIM].reshape(NUCLEUS_EFD_ORDER, 4)
    cell_contour = efd_decode(cell_coeffs, order=CELL_EFD_ORDER)
    nucleus_contour = efd_decode(nucleus_coeffs, order=NUCLEUS_EFD_ORDER)
    return cell_contour, nucleus_contour


def run_stage2(table, out_dir, k=PCA_K, qc_frames=9):
    n_frames = int(table["frame_idx"].max()) + 1
    video_id_list = sorted(np.unique(table["video_id"]).tolist())
    video_index = {vid: i for i, vid in enumerate(video_id_list)}
    n_videos = len(video_id_list)

    mean_arr = np.full((n_frames, CELL_DIM + NUCLEUS_DIM), np.nan)
    components_arr = np.full((n_frames, k, CELL_DIM + NUCLEUS_DIM), np.nan)
    evr_arr = np.full((n_frames, k), np.nan)
    k_used_arr = np.zeros(n_frames, dtype=int)
    n_valid_arr = np.zeros(n_frames, dtype=int)
    scores_arr = np.full((n_frames, n_videos, k), np.nan)

    results_by_frame = {}
    for t in range(n_frames):
        res = build_shape_space(table, t, k=k)
        results_by_frame[t] = res
        n_valid_arr[t] = res["n_valid"]
        k_used_arr[t] = res["k_used"]
        if res["k_used"] > 0:
            mean_arr[t] = res["mean"]
            components_arr[t, :res["k_used"]] = res["components"]
            evr_arr[t, :res["k_used"]] = res["explained_variance_ratio"]
            for vid, score_row in zip(res["video_ids"], res["scores"]):
                scores_arr[t, video_index[vid], :res["k_used"]] = score_row

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "shape_space.npz",
        mean=mean_arr, components=components_arr,
        explained_variance_ratio=evr_arr, k_used=k_used_arr, n_valid=n_valid_arr,
        scores=scores_arr, video_ids=np.array(video_id_list),
        frame_idx=np.arange(n_frames),
        cell_dim=CELL_DIM, nucleus_dim=NUCLEUS_DIM,
        order_cell=CELL_EFD_ORDER, order_nucleus=NUCLEUS_EFD_ORDER,
    )

    _print_stage2_summary(n_valid_arr, k_used_arr, evr_arr, n_videos)
    _plot_mean_shape_trajectory(results_by_frame, n_frames, out_dir, qc_frames, table=table)
    _plot_explained_variance(results_by_frame, n_frames, out_dir, qc_frames)

    return {
        "mean": mean_arr, "components": components_arr,
        "explained_variance_ratio": evr_arr, "k_used": k_used_arr,
        "n_valid": n_valid_arr, "scores": scores_arr, "video_ids": video_id_list,
    }


def _print_stage2_summary(n_valid_arr, k_used_arr, evr_arr, n_videos):
    print("\nStage 2 summary (pooled PCA, one per frame, across all videos):")
    print(f"  frames: {len(n_valid_arr)}, videos pooled: {n_videos}")
    print(f"  n_valid per frame: min={n_valid_arr.min()}, "
          f"mean={n_valid_arr.mean():.1f}, max={n_valid_arr.max()}")
    low_frames = np.flatnonzero(n_valid_arr < 0.5 * n_videos)
    if len(low_frames):
        print(f"  {len(low_frames)} frames with n_valid < 50% of videos "
              f"(likely division collapsing single-object frames), "
              f"e.g. frames {low_frames[:10].tolist()}")
    ok = k_used_arr > 0
    if ok.any():
        total_evr = np.nansum(evr_arr[ok], axis=1)
        print(f"  cumulative variance explained by top-{evr_arr.shape[1]} PCs: "
              f"min={total_evr.min():.2f}, mean={total_evr.mean():.2f}, max={total_evr.max():.2f}")
    n_skipped = int((~ok).sum())
    if n_skipped:
        print(f"  {n_skipped} frames skipped PCA entirely (n_valid < 3)")


def mean_nucleus2_contour(table, frame_idx, min_n=3):
    """Pooled mean shape of the SECOND (smaller) nucleus at `frame_idx`,
    across whichever videos have n_nuclei==2 there. This is a plain
    arithmetic mean (not a PCA fit -- nucleus_efd_2 was deliberately kept
    out of the Stage 2 PCA), which is exactly what pca.mean_ reduces to
    for the primary cell/nucleus anyway, so it's methodologically
    consistent with how those mean shapes are computed.

    Returns (contour, n) with contour=None if fewer than min_n videos have
    a second nucleus at this frame.
    """
    mask = (table["frame_idx"] == frame_idx) & table["valid_single"] & (table["n_nuclei"] == 2)
    n = int(mask.sum())
    if n < min_n:
        return None, n
    mean_vec = table["nucleus_efd_2"][mask].reshape(n, -1).mean(axis=0)
    coeffs = mean_vec.reshape(NUCLEUS_EFD_ORDER, 4)
    return efd_decode(coeffs, order=NUCLEUS_EFD_ORDER), n


def _plot_mean_shape_trajectory(results_by_frame, n_frames, out_dir, n_qc_frames, table=None, fps=12):
    """Animate the pooled mean cell+nucleus shape across every frame 0..n_frames-1
    into mean_shape_trajectory.gif (one axis range shared across all frames,
    same reasoning as the old static montage: per-frame auto-scaling would
    hide the actual size growth across the trajectory). When `table` is
    given, also overlays the pooled mean of the SECOND nucleus (post-
    mitosis, pre-cytokinesis frames) whenever enough videos have one at
    that frame.
    """
    contours_by_frame = {t: mean_shape_contours(results_by_frame[t]) for t in range(n_frames)}
    nucleus2_by_frame = (
        {t: mean_nucleus2_contour(table, t) for t in range(n_frames)}
        if table is not None else {t: (None, 0) for t in range(n_frames)}
    )
    all_pts = [c for pair in contours_by_frame.values() for c in pair if c is not None]
    all_pts += [c for c, _ in nucleus2_by_frame.values() if c is not None]
    if not all_pts:
        print("Mean shape trajectory: no PCA results available at any frame; skipping.")
        return
    all_xy = np.concatenate(all_pts, axis=0)
    x_range = all_xy[:, 0].max() - all_xy[:, 0].min()
    y_range = all_xy[:, 1].max() - all_xy[:, 1].min()
    margin = 0.08 * max(x_range, y_range)
    xlim = (all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ylim = (all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    fig, ax = plt.subplots(figsize=(5, 5))
    gif_frames = []
    for t in range(n_frames):
        ax.clear()
        cell_contour, nucleus_contour = contours_by_frame[t]
        nucleus2_contour, n_dividing = nucleus2_by_frame[t]
        if cell_contour is not None:
            cell_c = np.vstack([cell_contour, cell_contour[0]])
            nucleus_c = np.vstack([nucleus_contour, nucleus_contour[0]])
            ax.plot(cell_c[:, 0], cell_c[:, 1], "m-", lw=2, label="mean cell")
            ax.plot(nucleus_c[:, 0], nucleus_c[:, 1], "c-", lw=2, label="mean nucleus")
            if nucleus2_contour is not None:
                nucleus2_c = np.vstack([nucleus2_contour, nucleus2_contour[0]])
                ax.plot(nucleus2_c[:, 0], nucleus2_c[:, 1], "y-", lw=2,
                        label=f"mean nucleus #2 (n={n_dividing})")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no PCA\n(n_valid<3)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"frame {t} (n_valid={results_by_frame[t]['n_valid']})")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        gif_frames.append(Image.fromarray(buf).convert("RGB"))
    plt.close(fig)

    path = Path(out_dir) / "mean_shape_trajectory.gif"
    gif_frames[0].save(
        path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"  saved: {path} ({n_frames} frames, {fps} fps)")


def _plot_explained_variance(results_by_frame, n_frames, out_dir, n_qc_frames):
    frames = np.unique(np.linspace(0, n_frames - 1, min(n_qc_frames, 6)).astype(int))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in frames:
        res = results_by_frame[t]
        if res["k_used"] == 0:
            continue
        cum = np.cumsum(res["explained_variance_ratio"])
        ax.plot(np.arange(1, len(cum) + 1), cum, marker="o",
                label=f"frame {t} (n={res['n_valid']})")
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("cumulative explained variance ratio")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("Explained variance vs. #components, sampled frames")
    fig.tight_layout()
    path = Path(out_dir) / "explained_variance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=["nucleus", "random", "structure"],
                         help="Generation method to scope this run to; the 3 methods are "
                              "different perturbation conditions and are never pooled together.")
    parser.add_argument("--glob", default=None,
                         help="Override the full glob pattern (bypasses --method).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=PCA_K)
    parser.add_argument("--qc-samples", type=int, default=6)
    parser.add_argument("--qc-frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-normalize-rotation", action="store_true")
    parser.add_argument("--normalize-scale", action="store_true")
    args = parser.parse_args()

    pattern = args.glob if args.glob is not None else GLOB_TEMPLATE.format(method=args.method)

    table = run_stage1(
        pattern, args.out_dir, args.workers,
        normalize_rotation=not args.no_normalize_rotation,
        normalize_scale=args.normalize_scale,
        qc_samples=args.qc_samples, seed=args.seed,
    )
    run_stage2(table, args.out_dir, k=args.k, qc_frames=args.qc_frames)


if __name__ == "__main__":
    main()
