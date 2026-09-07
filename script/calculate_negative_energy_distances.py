import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from tqdm import tqdm

# ============================================
# ===== PORTABLE PATH & CONFIG SETTINGS =====
# ============================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# Automatically resolve paths relative to the project root
PROJECT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "results").exists() else SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FEATURES_CSV = RESULTS_DIR / "features_original_model_ready.csv"
DISPLACEMENTS_CSV = RESULTS_DIR / "cif_layer_displacements.csv"
OUTPUT_CSV = RESULTS_DIR / "negative_energy_geometry_distances.csv"

# Ensure output directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def wrap_positions_custom(positions, cell, pbc, center=(0.5, 0.5, 0.5)):
    """
    Consistent coordinate wrapping centered at 0.5 to keep layers 
    from splitting across boundaries during clustering.
    """
    inv_cell = np.linalg.inv(cell)
    frac = np.dot(positions, inv_cell)
    for i in range(3):
        if pbc[i]:
            shift = frac[:, i] - center[i] + 0.5
            frac[:, i] = (shift % 1.0) + center[i] - 0.5
    return np.dot(frac, cell)


def split_into_chains(symbols, positions):
    """
    Split the atoms of each element type into two halves based on Z-coordinates.
    Returns the coordinates of the lower and upper chains.
    """
    unique_elements = np.unique(symbols)
    lower_pos = []
    upper_pos = []
    for elem in unique_elements:
        mask = (symbols == elem)
        elem_pos = positions[mask]
        sort_idx = np.argsort(elem_pos[:, 2])
        sorted_pos = elem_pos[sort_idx]
        half = len(sorted_pos) // 2
        lower_pos.append(sorted_pos[:half])
        upper_pos.append(sorted_pos[half:])
    return np.vstack(lower_pos), np.vstack(upper_pos)


def calculate_centroids_distance(cif_path, select_elements=None):
    """
    Reads a CIF file, wraps positions, splits it into two chains, 
    and calculates the Euclidean distance between their geometric centers 
    accounting for periodic boundary conditions (Minimum Image Convention).
    """
    try:
        atoms = read(str(cif_path))
        cell = atoms.get_cell().array
        pbc = atoms.get_pbc()
        positions = atoms.get_positions()
        symbols = np.array(atoms.get_chemical_symbols())
        
        # Wrap positions
        positions_wrapped = wrap_positions_custom(positions, cell, pbc)
        
        # Filter elements if specified
        if select_elements is not None:
            mask = np.isin(symbols, select_elements)
            symbols = symbols[mask]
            positions_wrapped = positions_wrapped[mask]
            if len(positions_wrapped) == 0:
                return None
                
        lower_pos, upper_pos = split_into_chains(symbols, positions_wrapped)
        
        centroid_lower = lower_pos.mean(axis=0)
        centroid_upper = upper_pos.mean(axis=0)
        
        # Calculate PBC-aware distance vector
        inv_cell = np.linalg.inv(cell)
        c_lower_frac = centroid_lower @ inv_cell
        c_upper_frac = centroid_upper @ inv_cell
        delta_frac = c_upper_frac - c_lower_frac
        
        for i in range(3):
            if pbc[i]:
                delta_frac[i] -= np.round(delta_frac[i])
                
        delta_cart = delta_frac @ cell
        distance = np.linalg.norm(delta_cart)
        return float(distance)
    except Exception as e:
        print(f"[ERROR] Failed processing {cif_path}: {e}")
        return None


def main():
    print(f"Loading feature dataset from: {FEATURES_CSV}")
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing feature CSV: {FEATURES_CSV}")
    df_features = pd.read_csv(FEATURES_CSV)
    
    print(f"Loading displacement dataset from: {DISPLACEMENTS_CSV}")
    if not DISPLACEMENTS_CSV.exists():
        raise FileNotFoundError(f"Missing displacements CSV: {DISPLACEMENTS_CSV}")
    df_disp = pd.read_csv(DISPLACEMENTS_CSV)
    
    # Standardize paths to avoid slash/backslash mismatch issues
    df_features['normalized_path'] = df_features['file_path'].apply(lambda p: os.path.normpath(p).lower())
    df_disp['normalized_path'] = df_disp['file_path'].apply(lambda p: os.path.normpath(p).lower())
    
    # Merge datasets to have access to both energy and existing displacements
    df_merged = pd.merge(df_features, df_disp, on='normalized_path', suffixes=('_feat', '_disp'))
    
    # Filter for structures with negative energies (adjusted_energy < 0)
    df_negative = df_merged[df_merged['adjusted_energy'] < 0].copy()
    print(f"Found {len(df_negative)} structures with actual negative energy (adjusted_energy < 0).")
    
    results = []
    
    for idx, row in tqdm(df_negative.iterrows(), total=len(df_negative), desc="Calculating geometry center distances"):
        cif_path = row['file_path_feat']
        
        # 1. Compute geometry center distance for all atoms
        all_atoms_dist = calculate_centroids_distance(cif_path, select_elements=None)
        
        # 2. Compute geometry center distance for carbon atoms only
        carbon_dist = calculate_centroids_distance(cif_path, select_elements=['C'])
        
        # 3. Get existing displacement distance (Euclidean norm of X, Y, Z)
        X = row['X']
        Y = row['Y']
        Z = row['Z']
        displacement_dist = np.sqrt(X**2 + Y**2 + Z**2)
        
        results.append({
            'file_id': row['file_id'],
            'file_path': cif_path,
            'adjusted_energy': row['adjusted_energy'],
            'all_atoms_distance_mic': all_atoms_dist,
            'carbon_distance_mic': carbon_dist,
            'displacement_distance': displacement_dist,
            'X': X,
            'Y': Y,
            'Z': Z
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved geometry center distances to: {OUTPUT_CSV}")
    print(df_results.head(10))


if __name__ == "__main__":
    main()
