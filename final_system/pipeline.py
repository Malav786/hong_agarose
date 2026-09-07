from __future__ import annotations

import os
import re
import shutil
import json
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree as KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

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
def check_clash_pbc_jit(frac_orig, frac_new, cell, pbc, threshold_matrix):
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
            dist = np.sqrt(cx*cx + cy*cy + cz*cz)
            if dist < threshold_matrix[i, j] - 0.05:
                return True
    return False

@njit(cache=True, fastmath=True, nogil=True)
def get_min_dist_jit(frac_orig, frac_new, cell, pbc):
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
            dist = np.sqrt(cx*cx + cy*cy + cz*cz)
            if dist < min_dist:
                min_dist = dist
    return min_dist

# ================= HELPER GEOMETRY FUNCTIONS =================
Y_CUT = 0.5
Z_CUT = 0.8
VDW_RADII = {'C': 1.70, 'H': 1.20, 'O': 1.52}

def adjust_fractional(frac: np.ndarray, y_cut=Y_CUT, z_cut=Z_CUT) -> np.ndarray:
    f = frac.copy()
    f[:, 1] = np.where(f[:, 1] > y_cut, f[:, 1] - 1.0, f[:, 1])
    f[:, 2] = np.where(f[:, 2] > z_cut, f[:, 2] - 1.0, f[:, 2])
    return f

def rotate_about_x(points: np.ndarray, angle_deg: float, pivot: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    R = np.array([
        [1.0,         0.0,          0.0],
        [0.0,  np.cos(theta), -np.sin(theta)],
        [0.0,  np.sin(theta),  np.cos(theta)],
    ])
    shifted = points - pivot[None, :]
    return shifted @ R.T + pivot[None, :]

def wrap_positions_custom(positions, cell, pbc, center=(0.5, 0.5, 0.5)):
    inv_cell = np.linalg.inv(cell)
    frac = np.dot(positions, inv_cell)
    for i in range(3):
        if pbc[i]:
            shift = frac[:, i] - center[i] + 0.5
            frac[:, i] = (shift % 1.0) + center[i] - 0.5
    wrapped_pos = np.dot(frac, cell)
    return wrapped_pos

def load_and_unwrap_atoms(cif_path: str):
    atoms = read(cif_path)
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    pbc  = atoms.get_pbc()
    unwrapped = wrap_positions_custom(pos, cell, pbc, center=(0.5, 0.5, 0.5))
    atoms.set_positions(unwrapped)
    atoms.pbc = False
    return atoms

def distances_within(lower_coords: np.ndarray, upper_coords: np.ndarray, cutoff: float):
    if lower_coords.shape[0] == 0 or upper_coords.shape[0] == 0:
        return []
    tree = KDTree(lower_coords)
    dists, _ = tree.query(upper_coords, k=1)
    dists = np.asarray(dists).ravel()
    return dists[dists <= cutoff].tolist()

def select_grids_radius_prune(df_pts: pd.DataFrame, target_n: int = 100, radius: float = 4.0, seed: int | None = None) -> pd.DataFrame:
    pts = df_pts[["gx", "gy", "gz"]].to_numpy(float)
    if pts.shape[0] == 0:
        return df_pts.iloc[0:0].copy()
    remaining = list(range(pts.shape[0]))
    selected = []
    rng = np.random.default_rng(seed)
    while remaining and len(selected) < target_n:
        pick_pos = int(rng.integers(0, len(remaining)))
        pick_idx = remaining[pick_pos]
        pick_pt = pts[pick_idx][None, :]
        rem_pts = pts[np.array(remaining)]
        nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree")
        nbrs.fit(rem_pts)
        neigh_pos = nbrs.radius_neighbors(pick_pt, return_distance=False)[0].tolist()
        neigh_global = [remaining[p] for p in neigh_pos]
        selected.append(pick_idx)
        remove_set = set(neigh_global)
        remaining = [i for i in remaining if i not in remove_set]
    return df_pts.iloc[selected].reset_index(drop=True)

# ================= STAGE 1: N-LAYER STRUCTURE GENERATION =================
def extract_r_from_path(path: str) -> int:
    norm = path.replace("\\", "/")
    m = re.search(r"/r(\d+)(?:/|$)", norm)
    return int(m.group(1)) if m else 0

def extract_t_from_path(path: str) -> str:
    norm = path.replace("\\", "/")
    m = re.search(r"/t([0-9.]+)/(?:[^/]+)$", norm)
    return m.group(1) if m else "0"

def extract_angle_from_filename(path: str) -> float:
    fname = os.path.basename(path)
    m = re.search(r"_([0-9]+)\.cif$", fname)
    return float(m.group(1)) if m else 0.0

def generate_next_layer(cif_path: str, out_root: str, chosen_angles: np.ndarray, L: int, logger=print):
    """
    Takes an (L-1)-layer structure, stacks an L-th layer below (lower) and above (upper).
    """
    rnum = extract_r_from_path(cif_path)
    disp_val_str = extract_t_from_path(cif_path)
    disp_val = float(disp_val_str)
    base_upper_angle = extract_angle_from_filename(cif_path)

    # Read CIF file
    structure = read(cif_path)
    frac = structure.get_scaled_positions()
    structure.set_scaled_positions(adjust_fractional(frac))
    positions = structure.get_positions()
    cell = structure.get_cell()
    cell_arr = np.array(cell)
    pbc = structure.get_pbc()
    symbols = np.array(structure.get_chemical_symbols())

    df_atoms = pd.DataFrame(positions, columns=["x", "y", "z"])
    df_atoms.insert(0, "Element", symbols)

    # Sort all atoms by Z-coordinate to identify existing layers
    df_sorted = df_atoms.sort_values(by="z").reset_index(drop=True)
    expected_atoms = (L - 1) * 78
    if len(df_sorted) != expected_atoms:
        logger(f"Skip {cif_path}: expected {expected_atoms} atoms, got {len(df_sorted)}")
        return

    # Identify individual layers (each layer is exactly 78 atoms)
    layers = [df_sorted.iloc[i*78 : (i+1)*78].copy().reset_index(drop=True) for i in range(L-1)]
    bottom_layer = layers[0]
    top_layer = layers[-1]

    # Measure separation bounds from adjacent interfaces
    # We measure separation of the closest interface:
    # If L-1 = 2 (bilayer), we measure interface 1 (between layer 1 and 2).
    # In general, we measure the separation between layers L-2 and L-1.
    if L - 1 >= 2:
        # Compute adjacent interface distance to match separation
        l_adj = layers[-2]
        u_adj = layers[-1]
        c_l_adj = np.concatenate([l_adj[l_adj["Element"] == "C"][["x","y","z"]].values, l_adj[l_adj["Element"] == "O"][["x","y","z"]].values]).mean(axis=0)
        c_u_adj = np.concatenate([u_adj[u_adj["Element"] == "C"][["x","y","z"]].values, u_adj[u_adj["Element"] == "O"][["x","y","z"]].values]).mean(axis=0)
        D_bilayer = np.linalg.norm(c_u_adj - c_l_adj)
    else:
        D_bilayer = 7.58  # Bilayer dataset average separation fallback

    TM_MIN = D_bilayer - 0.5
    TM_MAX = D_bilayer + 0.5

    # Identify Centroids for Lower and Upper layers
    c_l = np.concatenate([bottom_layer[bottom_layer["Element"] == "C"][["x","y","z"]].values, bottom_layer[bottom_layer["Element"] == "O"][["x","y","z"]].values]).mean(axis=0)
    c_u = np.concatenate([top_layer[top_layer["Element"] == "C"][["x","y","z"]].values, top_layer[top_layer["Element"] == "O"][["x","y","z"]].values]).mean(axis=0)

    # ---------------- ELEMENT-WISE PAIR THRESHOLDS ----------------
    # Compute PBC-aware pairwise distances between adjacent layers to extract threshold
    inv_cell = np.linalg.inv(cell_arr)
    
    # We use bottom layer to extract element-wise separation thresholds
    lower_pos = np.concatenate([bottom_layer[bottom_layer["Element"] == "C"][["x","y","z"]].values, 
                                bottom_layer[bottom_layer["Element"] == "H"][["x","y","z"]].values, 
                                bottom_layer[bottom_layer["Element"] == "O"][["x","y","z"]].values])
    lower_elems = np.array(['C']*len(bottom_layer[bottom_layer["Element"]=="C"]) + 
                           ['H']*len(bottom_layer[bottom_layer["Element"]=="H"]) + 
                           ['O']*len(bottom_layer[bottom_layer["Element"]=="O"]))

    second_layer = layers[1] if L-1 >= 2 else layers[0]
    upper_pos = np.concatenate([second_layer[second_layer["Element"] == "C"][["x","y","z"]].values, 
                                second_layer[second_layer["Element"] == "H"][["x","y","z"]].values, 
                                second_layer[second_layer["Element"] == "O"][["x","y","z"]].values])
    upper_elems = np.array(['C']*len(second_layer[second_layer["Element"]=="C"]) + 
                           ['H']*len(second_layer[second_layer["Element"]=="H"]) + 
                           ['O']*len(second_layer[second_layer["Element"]=="O"]))

    frac_l = lower_pos @ inv_cell
    frac_u = upper_pos @ inv_cell
    diff_frac = frac_l[:, None, :] - frac_u[None, :, :]
    for i in range(3):
        if pbc[i]:
            diff_frac[:, :, i] -= np.round(diff_frac[:, :, i])
    parent_dists = np.linalg.norm(diff_frac @ cell_arr, axis=2)
    d_min_parent = parent_dists.min()

    unique_elements = ['C', 'H', 'O']
    d_min_parent_pairs = {}
    for el1 in unique_elements:
        for el2 in unique_elements:
            pair_key = tuple(sorted([el1, el2]))
            idx_l = np.where(lower_elems == el1)[0]
            idx_u = np.where(upper_elems == el2)[0]
            d_min = 99.0
            if len(idx_l) > 0 and len(idx_u) > 0:
                d_min = min(d_min, parent_dists[idx_l[:, None], idx_u[None, :]].min())
            idx_l2 = np.where(lower_elems == el2)[0]
            idx_u2 = np.where(upper_elems == el1)[0]
            if len(idx_l2) > 0 and len(idx_u2) > 0:
                d_min = min(d_min, parent_dists[idx_l2[:, None], idx_u2[None, :]].min())
            if d_min > 50.0:
                d_min = 0.7 * (VDW_RADII[el1] + VDW_RADII[el2])
            d_min_parent_pairs[pair_key] = d_min

    # Create Threshold Matrix (against the entire parent structure)
    orig_all_elems = df_sorted["Element"].to_numpy()
    orig_all_xyz = df_sorted[["x","y","z"]].to_numpy(float)
    threshold_matrix = np.zeros((len(orig_all_elems), 78))
    for i, e_f in enumerate(orig_all_elems):
        for j in range(78):
            template_e = bottom_layer["Element"].iloc[j]
            pair_key = tuple(sorted([e_f, template_e]))
            threshold_matrix[i, j] = d_min_parent_pairs[pair_key]

    # Generate Grid search coordinates
    X_CART = 9.35
    Y_grid = np.arange(0, 24.0, 0.5)
    Z_grid = np.arange(0, 24.0, 0.5)
    coords = [(X_CART, y, z) for y in Y_grid for z in Z_grid]
    df_grid = pd.DataFrame(coords, columns=["x", "y", "z"])
    grid_xyz = df_grid[["x", "y", "z"]].to_numpy(float)

    d_l = np.linalg.norm(grid_xyz - c_l[None, :], axis=1)
    d_u = np.linalg.norm(grid_xyz - c_u[None, :], axis=1)

    df_vecs = pd.DataFrame({
        "gx": grid_xyz[:, 0], "gy": grid_xyz[:, 1], "gz": grid_xyz[:, 2],
        "d_l": d_l, "d_u": d_u
    })

    in_range_U = df_vecs["d_u"].between(TM_MIN, TM_MAX) & ~df_vecs["d_l"].between(0, TM_MIN)
    in_range_L = df_vecs["d_l"].between(TM_MIN, TM_MAX) & ~df_vecs["d_u"].between(0, TM_MIN)
    df_in_range = df_vecs.loc[in_range_U | in_range_L].copy()
    df_in_range = df_in_range.loc[df_in_range["d_l"] != df_in_range["d_u"]].copy()
    df_in_range["label"] = 0
    df_in_range.loc[df_in_range["d_u"] < df_in_range["d_l"], "label"] = 1

    df_lower_grids = df_in_range.loc[df_in_range["label"] == 0, ["gx","gy","gz"]].reset_index(drop=True)
    df_upper_grids = df_in_range.loc[df_in_range["label"] == 1, ["gx","gy","gz"]].reset_index(drop=True)

    df_lower_grids = select_grids_radius_prune(df_lower_grids, target_n=100, radius=4.0)
    df_upper_grids = select_grids_radius_prune(df_upper_grids, target_n=100, radius=4.0)

    # Stacking logic
    def write_for_side(kind: str, grid_idx: int, pivot: np.ndarray, origin_df: pd.DataFrame, neutral_angle_deg: float):
        new_chain = origin_df.copy()
        centroid_new = new_chain[["x","y","z"]].mean(axis=0).to_numpy()
        new_chain[["x","y","z"]] = new_chain[["x","y","z"]] + (pivot - centroid_new)

        new_xyz = new_chain[["x","y","z"]].to_numpy(float)
        new_elems = new_chain["Element"].to_numpy()

        neutral_xyz = rotate_about_x(new_xyz, neutral_angle_deg, pivot)
        disp_vec = np.array([disp_val, 0.0, 0.0])
        displaced_xyz = neutral_xyz + disp_vec[None, :]
        pivot_disp = pivot + disp_vec

        ref_centroid = c_l if kind == "lower" else c_u
        vec = pivot_disp - ref_centroid
        u = vec / np.linalg.norm(vec)

        for ang in chosen_angles:
            rotated_xyz = rotate_about_x(displaced_xyz, -ang, pivot_disp)
            frac_orig = orig_all_xyz @ inv_cell

            # Bisection optimized exact gap alignment
            low, high = -2.5, 2.5
            for _ in range(12):
                mid = (low + high) / 2.0
                shifted_xyz = rotated_xyz + mid * u
                frac_new = shifted_xyz @ inv_cell
                d_mid = get_min_dist_jit(frac_orig, frac_new, cell_arr, pbc)
                if d_mid < d_min_parent:
                    low = mid
                else:
                    high = mid
            s_opt = (low + high) / 2.0
            optimized_xyz = rotated_xyz + s_opt * u

            # Clash Check
            frac_opt = optimized_xyz @ inv_cell
            if check_clash_pbc_jit(frac_orig, frac_opt, cell_arr, pbc, threshold_matrix):
                continue

            merged_elems = np.concatenate([orig_all_elems, new_elems])
            merged_xyz = np.vstack([orig_all_xyz, optimized_xyz])

            atoms_out = Atoms(symbols=merged_elems, positions=merged_xyz, cell=cell, pbc=True)

            # ================= DYNAMIC CELL ENLARGEMENT =================
            orig_cell_arr = atoms_out.get_cell().array
            enlarged_cell = orig_cell_arr.copy()

            # Add 50 Å to the current b-vector length (keep direction)
            b_vec = enlarged_cell[1]
            b_len = np.linalg.norm(b_vec)
            if b_len > 1e-8:
                new_b_len = b_len + 50.0
                enlarged_cell[1] = b_vec * (new_b_len / b_len)
            else:
                enlarged_cell[1] = np.array([0.0, 50.0, 0.0])

            # Add 50 Å to the current c-vector length (keep direction)
            c_vec = enlarged_cell[2]
            c_len = np.linalg.norm(c_vec)
            if c_len > 1e-8:
                new_c_len = c_len + 50.0
                enlarged_cell[2] = c_vec * (new_c_len / c_len)
            else:
                enlarged_cell[2] = np.array([0.0, 0.0, 50.0])

            atoms_out.set_cell(enlarged_cell, scale_atoms=False)
            # ============================================================

            out_dir = os.path.join(out_root, kind, f"r{rnum}", f"t{disp_val_str}")
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"t{disp_val_str}_{ang}_grid{grid_idx}.cif"
            atoms_out.write(os.path.join(out_dir, out_name))

    # Stacking below bottom layer
    base_lower_angle = -float(rnum)
    for gi, row in df_lower_grids.iterrows():
        pivot = row.to_numpy(float)
        write_for_side("lower", gi, pivot, bottom_layer, base_lower_angle)

    # Stacking above top layer
    for gi, row in df_upper_grids.iterrows():
        pivot = row.to_numpy(float)
        write_for_side("upper", gi, pivot, top_layer, base_upper_angle)

def run_n_layer_generation(L: int, base_in: str, out_root: str, chosen_angles: np.ndarray, cores: int, logger=print):
    all_cifs = []
    for root, _, files in os.walk(base_in):
        for f in files:
            if f.lower().endswith(".cif"):
                all_cifs.append(os.path.join(root, f))
    
    logger(f"Total parent CIF files found: {len(all_cifs)}")
    logger(f"Running N-layer generation for L={L} using {cores} worker threads...")

    tasks = [(cif, out_root, chosen_angles, L) for cif in all_cifs]
    
    completed = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(generate_next_layer, cif, out, angs, l, logger): cif for cif, out, angs, l in tasks}
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            pct = int((completed / total) * 100)
            if completed % max(1, total // 20) == 0 or completed == total:
                logger(f"Generating {L}-layer configurations: {pct}% completed ({completed}/{total})")
    logger(f"Finished generating candidates for L={L}")

# ================= STAGE 2: MULTI-INTERFACE FEATURE EXTRACTION =================
def compute_bounds_for_n_layer(cif_path: str, L: int, rvw_H=1.20, rvw_O=1.53, rvw_err=0.10) -> dict:
    try:
        atoms = load_and_unwrap_atoms(cif_path)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        df = pd.DataFrame(positions, columns=["x", "y", "z"])
        df.insert(0, "Element", symbols)

        # Sort along Z
        df_sorted = df.sort_values(by="z").reset_index(drop=True)
        expected_atoms = L * 78
        if len(df_sorted) != expected_atoms:
            raise ValueError(f"Expected {expected_atoms} atoms for L={L}, found {len(df_sorted)}")

        # Split into individual layers (78 atoms each)
        layers = [df_sorted.iloc[i*78 : (i+1)*78].copy().reset_index(drop=True) for i in range(L)]

        cut_HH = 2 * rvw_H + rvw_err
        cut_OO = 2 * rvw_O + rvw_err
        cut_OH = rvw_O + rvw_H + rvw_err

        res = {"status": "success", "file_path": cif_path}

        # Compute features for each adjacent interface (Layer i vs Layer i+1)
        for i in range(L - 1):
            l_layer = layers[i]
            u_layer = layers[i+1]

            lH_c = coords_of_df(l_layer[l_layer["Element"] == "H"])
            lO_c = coords_of_df(l_layer[l_layer["Element"] == "O"])

            uH_c = coords_of_df(u_layer[u_layer["Element"] == "H"])
            uO_c = coords_of_df(u_layer[u_layer["Element"] == "O"])

            dist_HH = distances_within(lH_c, uH_c, cut_HH)
            dist_OO = distances_within(lO_c, uO_c, cut_OO)
            dist_OH = distances_within(lH_c, uO_c, cut_OH)
            dist_HO = distances_within(lO_c, uH_c, cut_OH)

            # Namespace interface features: Int1_*, Int2_*, etc.
            prefix = f"Int{i+1}"
            res[f"{prefix}_avg_HH_dist"] = mean_or_default(dist_HH)
            res[f"{prefix}_avg_OO_dist"] = mean_or_default(dist_OO)
            res[f"{prefix}_avg_OH_dist"] = mean_or_default(dist_OH)
            res[f"{prefix}_avg_HO_dist"] = mean_or_default(dist_HO)
            res[f"{prefix}_count_HH"] = len(dist_HH)
            res[f"{prefix}_count_OO"] = len(dist_OO)
            res[f"{prefix}_count_OH"] = len(dist_OH)
            res[f"{prefix}_count_HO"] = len(dist_HO)
            res[f"{prefix}_sum_of_count"] = len(dist_HH) + len(dist_OO) + len(dist_OH) + len(dist_HO)

        return res
    except Exception as e:
        return {"status": "error", "file_path": cif_path, "error": str(e)}

def mean_or_default(values, default=3.0) -> float:
    return float(np.mean(values)) if len(values) else float(default)

def coords_of_df(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 3), dtype=float)
    return df[["x", "y", "z"]].to_numpy(dtype=float)

def path_sort_key_generalized(path):
    path_str = str(path).replace('\\', '/')
    kind = "lower" if "/lower/" in path_str else "upper"
    rot_match = re.search(r'/r(\d+)/', path_str)
    rot_val = int(rot_match.group(1)) if rot_match else 0
    disp_match = re.search(r'/t(\d+(?:\.\d+)?)/', path_str)
    disp_val = float(disp_match.group(1)) if disp_match else 0.0
    fname = os.path.basename(path_str)
    angle_match = re.search(r'_(\d+)_grid(\d+)\.cif$', fname)
    if angle_match:
        angle_val = int(angle_match.group(1))
        grid_val = int(angle_match.group(2))
    else:
        angle_val, grid_val = 0, 0
    return (kind, rot_val, disp_val, angle_val, grid_val)

def run_n_layer_extraction(L: int, base_dir: str, out_csv: str, cores: int, logger=print):
    cif_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".cif"):
                cif_paths.append(os.path.join(root, file))
    
    total_files = len(cif_paths)
    logger(f"Extracting features from {total_files} CIF configurations for L={L}...")

    all_results = []
    errors = []
    completed = 0
    with ThreadPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(compute_bounds_for_n_layer, p, L): p for p in cif_paths}
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success":
                all_results.append(res)
            else:
                errors.append((res["file_path"], res["error"]))
            
            completed += 1
            pct = int((completed / total_files) * 100)
            if completed % max(1, total_files // 20) == 0 or completed == total_files:
                logger(f"Extracting {L}-layer bounds features: {pct}% completed ({completed}/{total_files})")

    df_all = pd.DataFrame(all_results)
    if df_all.empty:
        logger("No features extracted.")
        return

    df_all.drop(columns=["status"], errors="ignore", inplace=True)
    df_all["sort_key"] = df_all["file_path"].apply(path_sort_key_generalized)
    df_all = df_all.sort_values(by="sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    cols_feature = [c for c in df_all.columns if c not in ("file_path", "status")]
    df_final = df_all[["file_path"] + cols_feature]
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    logger(f"Feature extraction complete! Saved to: {out_csv}")
    if errors:
        logger(f"Encountered {len(errors)} errors during extraction. First 5 shown:")
        for path, msg in errors[:5]:
            logger(f"- {path}: {msg}")

# ================= STAGE 3: N-INTERFACE CLASSIFIER INFERENCE =================
def run_n_layer_predictions(L: int, csv_path: str, model_path: str, metadata_path: str, dest_dir: str, source_dir: str, logger=print):
    logger(f"Loading predictions dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature CSV not found at {csv_path}")

    df_full = pd.read_csv(csv_path)

    # Load Model and Config
    with open(metadata_path, "r") as f:
        config = json.load(f)
    threshold = float(config["threshold"])
    feature_names = list(config["features"])

    import joblib
    model = joblib.load(model_path)
    logger(f"Loaded ML pipeline model from: {model_path}")

    # Predict stability for each of the L-1 interfaces
    interface_preds = []
    
    for i in range(L - 1):
        prefix = f"Int{i+1}"
        int_cols = [f"{prefix}_{col}" for col in feature_names]
        
        X_int = df_full[int_cols].copy()
        X_int.columns = feature_names
        
        probs = model.predict_proba(X_int)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        interface_preds.append(preds)
        logger(f"Interface {i+1} Class 0 (Stable): {(preds == 0).sum()} | Class 1 (Unstable): {(preds == 1).sum()}")

    # Consensus prediction: stable if ALL interfaces are stable (Class 0)
    final_preds = np.zeros(len(df_full), dtype=int)
    for preds in interface_preds:
        final_preds = np.maximum(final_preds, preds)  # if any interface is 1, final is 1

    stable_mask = (final_preds == 0)
    total_stable = stable_mask.sum()
    logger(f"Overall {L}-layer Class 0 (Stable): {total_stable} | Class 1 (Unstable): {len(df_full) - total_stable}")

    # Copy stable structures
    if os.path.exists(dest_dir):
        logger(f"Removing existing destination folder: {dest_dir}")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0
    stable_paths = df_full.loc[stable_mask, "file_path"].tolist()

    for orig_path in stable_paths:
        rel_path = os.path.relpath(orig_path, source_dir)
        if rel_path.startswith(".."):
            parts = orig_path.replace("\\", "/").split(f"/{L}_layer/")
            rel_path = parts[-1] if len(parts) > 1 else os.path.basename(orig_path)
            
        src = os.path.join(source_dir, rel_path)
        dest = os.path.join(dest_dir, rel_path)
        
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            copied_count += 1
        else:
            missing_count += 1

    logger("\n================ COPY SUMMARY ================")
    logger(f"Total Predicted Class 0 (Stable): {total_stable}")
    logger(f"Successfully copied             : {copied_count} files")
    logger(f"Missing source files            : {missing_count}")
    logger(f"Saved into                      : {dest_dir}")
    logger("==============================================")
