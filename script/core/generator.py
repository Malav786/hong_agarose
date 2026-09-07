"""
High-Performance N-Layer Candidate Generation Engine.
Features Numba JIT-accelerated clash detection with GIL release,
PBC-aware gap bisection alignment, out-of-plane vacuum padding,
and atomic file writes to prevent CIF file corruption under multithreading.
"""

from __future__ import annotations

import os
import re
import tempfile
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .geometry import (
    validate_structure,
    wrap_positions_custom,
    partition_layers_dynamically,
    calculate_layer_centroids,
    VDW_RADII
)

# ================= GRACEFUL NUMBA JIT DECORATOR =================
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


# ================= JIT COMPRESSED MATH OPERATIONS (GIL Released) =================
@njit(cache=True, fastmath=True, nogil=True)
def check_clash_pbc_jit(
    frac_orig: np.ndarray,
    frac_new: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    threshold_matrix: np.ndarray,
    buffer: float = 0.05
) -> bool:
    """
    Checks for steric clashes between original and candidate atom positions.
    Exits early on the very first clash detected.
    """
    n_orig = frac_orig.shape[0]
    n_new = frac_new.shape[0]

    for i in range(n_orig):
        for j in range(n_new):
            dx = frac_orig[i, 0] - frac_new[j, 0]
            dy = frac_orig[i, 1] - frac_new[j, 1]
            dz = frac_orig[i, 2] - frac_new[j, 2]

            if pbc[0]:
                dx -= np.round(dx)
            if pbc[1]:
                dy -= np.round(dy)
            if pbc[2]:
                dz -= np.round(dz)

            cx = dx * cell[0, 0] + dy * cell[1, 0] + dz * cell[2, 0]
            cy = dx * cell[0, 1] + dy * cell[1, 1] + dz * cell[2, 1]
            cz = dx * cell[0, 2] + dy * cell[1, 2] + dz * cell[2, 2]

            dist = np.sqrt(cx * cx + cy * cy + cz * cz)
            if dist < threshold_matrix[i, j] - buffer:
                return True
    return False


@njit(cache=True, fastmath=True, nogil=True)
def get_min_dist_jit(
    frac_orig: np.ndarray,
    frac_new: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray
) -> float:
    """Computes exact minimum interatomic distance under PBC without allocating arrays."""
    n_orig = frac_orig.shape[0]
    n_new = frac_new.shape[0]
    min_dist = 9999.0

    for i in range(n_orig):
        for j in range(n_new):
            dx = frac_orig[i, 0] - frac_new[j, 0]
            dy = frac_orig[i, 1] - frac_new[j, 1]
            dz = frac_orig[i, 2] - frac_new[j, 2]

            if pbc[0]:
                dx -= np.round(dx)
            if pbc[1]:
                dy -= np.round(dy)
            if pbc[2]:
                dz -= np.round(dz)

            cx = dx * cell[0, 0] + dy * cell[1, 0] + dz * cell[2, 0]
            cy = dx * cell[0, 1] + dy * cell[1, 1] + dz * cell[2, 1]
            cz = dx * cell[0, 2] + dy * cell[1, 2] + dz * cell[2, 2]

            dist = np.sqrt(cx * cx + cy * cy + cz * cz)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def rotate_about_x(points: np.ndarray, angle_deg: float, pivot: np.ndarray) -> np.ndarray:
    """Rotates 3D points about the X-axis passing through a pivot point."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ])
    return (points - pivot[None, :]) @ R.T + pivot[None, :]


def atomic_write_cif(atoms: Atoms, target_path: str) -> None:
    """
    Safely writes an Atoms object to disk using atomic rename.
    Guarantees that parallel processes/threads NEVER leave half-written or corrupted CIF files.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temporary file in the same directory first
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".cif.tmp", dir=str(target.parent))
    os.close(tmp_fd)
    
    try:
        write(tmp_name, atoms, format="cif")
        # Atomic rename on POSIX and modern Windows (Python 3.3+)
        os.replace(tmp_name, str(target))
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def compute_element_threshold_matrix(
    parent_atoms_df: pd.DataFrame,
    template_layer_df: pd.DataFrame,
    cell: np.ndarray,
    pbc: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Extracts element-pair minimum distance thresholds observed in the parent configuration.
    Falls back to 0.7 * (vdW1 + vdW2) for unseen element pairs.
    """
    inv_cell = np.linalg.inv(cell)
    pos_parent = parent_atoms_df[["x", "y", "z"]].to_numpy(float)
    elems_parent = parent_atoms_df["Element"].to_numpy()
    
    pos_tmpl = template_layer_df[["x", "y", "z"]].to_numpy(float)
    elems_tmpl = template_layer_df["Element"].to_numpy()
    
    # Compute pairwise distances under PBC
    diff_frac = (pos_tmpl @ inv_cell)[None, :, :] - (pos_parent @ inv_cell)[:, None, :]
    for i in range(3):
        if pbc[i]:
            diff_frac[:, :, i] -= np.round(diff_frac[:, :, i])
    parent_dists = np.linalg.norm(diff_frac @ cell, axis=2)
    d_min_observed = float(parent_dists.min()) if parent_dists.size else 2.0
    
    unique_elements = ['C', 'H', 'O']
    pair_thresholds: Dict[Tuple[str, str], float] = {}
    
    for el1 in unique_elements:
        for el2 in unique_elements:
            key = tuple(sorted([el1, el2]))
            idx_p = np.where(elems_parent == el1)[0]
            idx_t = np.where(elems_tmpl == el2)[0]
            d_min = 99.0
            if len(idx_p) > 0 and len(idx_t) > 0:
                d_min = min(d_min, parent_dists[idx_p[:, None], idx_t[None, :]].min())
            if d_min > 50.0:
                d_min = 0.7 * (VDW_RADII.get(el1, 1.5) + VDW_RADII.get(el2, 1.5))
            pair_thresholds[key] = d_min

    # Build threshold matrix
    threshold_matrix = np.zeros((len(elems_parent), len(elems_tmpl)))
    for i, e_p in enumerate(elems_parent):
        for j, e_t in enumerate(elems_tmpl):
            threshold_matrix[i, j] = pair_thresholds[tuple(sorted([e_p, e_t]))]
            
    return threshold_matrix, d_min_observed


def generate_candidates_for_parent(
    parent_cif_path: str,
    out_root: str,
    chosen_angles: np.ndarray,
    L: int,
    equilibrium_gap: float = 7.58,
    gap_tolerance: float = 0.50
) -> int:
    """
    Takes an (L - 1)-layer parent structure, stacks a new monolayer on both sides (lower and upper),
    and generates candidate structures with atomic clash checks and gap optimization.
    
    Returns:
        Number of valid candidates successfully written.
    """
    atoms = read(parent_cif_path)
    validate_structure(atoms)

    cell = atoms.get_cell().array
    pbc = atoms.get_pbc()
    inv_cell = np.linalg.inv(cell)

    # Dynamic monolayer partitioning
    layers = partition_layers_dynamically(atoms, L - 1)
    atoms_per_layer = len(layers[0])
    
    bottom_layer = layers[0]
    top_layer = layers[-1]

    c_bottom = calculate_layer_centroids(bottom_layer)
    c_top = calculate_layer_centroids(top_layer)

    # Measure observed separation between adjacent layers
    if L - 1 >= 2:
        d_obs = np.linalg.norm(calculate_layer_centroids(layers[-1]) - calculate_layer_centroids(layers[-2]))
    else:
        d_obs = equilibrium_gap

    tm_min = d_obs - gap_tolerance
    tm_max = d_obs + gap_tolerance

    # Build clash threshold matrix
    all_atoms_df = pd.concat(layers, ignore_index=True)
    threshold_matrix, d_min_parent = compute_element_threshold_matrix(
        all_atoms_df, bottom_layer, cell, pbc
    )

    parent_elems = all_atoms_df["Element"].to_numpy()
    parent_xyz = all_atoms_df[["x", "y", "z"]].to_numpy(float)
    frac_orig = parent_xyz @ inv_cell

    # Extract rotation and displacement info from path
    norm_path = parent_cif_path.replace("\\", "/")
    r_match = re.search(r'/r(\d+)/', norm_path)
    rnum = int(r_match.group(1)) if r_match else 0
    t_match = re.search(r'/t([0-9.]+)/', norm_path)
    disp_val_str = t_match.group(1) if t_match else "0"
    disp_val = float(disp_val_str)

    # Generate sampling grid around centroids
    x_cart = 9.35
    y_grid = np.arange(0, 24.0, 0.75)
    z_grid = np.arange(0, 24.0, 0.75)
    coords = np.array([(x_cart, y, z) for y in y_grid for z in z_grid], dtype=float)

    d_l = np.linalg.norm(coords - c_bottom[None, :], axis=1)
    d_u = np.linalg.norm(coords - c_top[None, :], axis=1)

    lower_mask = (d_l >= tm_min) & (d_l <= tm_max) & (d_l < d_u)
    upper_mask = (d_u >= tm_min) & (d_u <= tm_max) & (d_u < d_l)

    lower_pivots = coords[lower_mask][:40]  # sample subset
    upper_pivots = coords[upper_mask][:40]

    written_count = 0

    def stack_and_save(kind: str, grid_idx: int, pivot: np.ndarray, template_df: pd.DataFrame, base_angle: float):
        nonlocal written_count
        new_chain = template_df.copy()
        c_template = new_chain[["x", "y", "z"]].mean(axis=0).to_numpy()
        new_chain[["x", "y", "z"]] += (pivot - c_template)

        raw_xyz = new_chain[["x", "y", "z"]].to_numpy(float)
        new_elems = new_chain["Element"].to_numpy()

        neutral_xyz = rotate_about_x(raw_xyz, base_angle, pivot)
        disp_vec = np.array([disp_val, 0.0, 0.0])
        displaced_xyz = neutral_xyz + disp_vec[None, :]
        pivot_disp = pivot + disp_vec

        ref_centroid = c_bottom if kind == "lower" else c_top
        u_vec = pivot_disp - ref_centroid
        u_norm = np.linalg.norm(u_vec)
        if u_norm < 1e-6:
            return
        u = u_vec / u_norm

        for ang in chosen_angles:
            rotated_xyz = rotate_about_x(displaced_xyz, -ang, pivot_disp)

            # Bisection alignment to match adjacent gap distance
            low, high = -2.5, 2.5
            for _ in range(12):
                mid = (low + high) / 2.0
                shifted = rotated_xyz + mid * u
                frac_candidate = shifted @ inv_cell
                d_mid = get_min_dist_jit(frac_orig, frac_candidate, cell, pbc)
                if d_mid < d_min_parent:
                    low = mid
                else:
                    high = mid
            s_opt = (low + high) / 2.0
            optimized_xyz = rotated_xyz + s_opt * u

            # Clash Check
            frac_opt = optimized_xyz @ inv_cell
            if check_clash_pbc_jit(frac_orig, frac_opt, cell, pbc, threshold_matrix):
                continue

            merged_elems = np.concatenate([parent_elems, new_elems])
            merged_xyz = np.vstack([parent_xyz, optimized_xyz])

            # Build atoms object with out-of-plane vacuum padding
            atoms_out = Atoms(symbols=merged_elems, positions=merged_xyz, cell=cell, pbc=True)
            
            # Preserve in-plane periodicity, pad out-of-plane stacking axis
            padded_cell = cell.copy()
            c_vec = padded_cell[2]
            c_len = np.linalg.norm(c_vec)
            if c_len > 1e-6:
                padded_cell[2] = c_vec * ((c_len + 30.0) / c_len)
            atoms_out.set_cell(padded_cell, scale_atoms=False)

            out_dir = os.path.join(out_root, kind, f"r{rnum}", f"t{disp_val_str}")
            out_name = f"t{disp_val_str}_{ang}_grid{grid_idx}.cif"
            out_path = os.path.join(out_dir, out_name)

            atomic_write_cif(atoms_out, out_path)
            written_count += 1

    # Stacking below
    for gi, pivot in enumerate(lower_pivots):
        stack_and_save("lower", gi, pivot, bottom_layer, -float(rnum))

    # Stacking above
    for gi, pivot in enumerate(upper_pivots):
        stack_and_save("upper", gi, pivot, top_layer, 0.0)

    return written_count
