"""
Calculates the geometric centroid distance between layers for stable (negative energy) structures
accounting for Periodic Boundary Conditions under Minimum Image Convention (MIC).
Refactored to utilize the robust core.geometry engine.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from tqdm import tqdm

# Add parent directory to path to enable core package import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.geometry import (
    validate_structure,
    wrap_positions_custom,
    partition_layers_dynamically,
    calculate_layer_centroids,
    min_image_displacement
)

PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FEATURES_CSV = RESULTS_DIR / "features_original_model_ready.csv"
DISPLACEMENTS_CSV = RESULTS_DIR / "cif_layer_displacements.csv"
OUTPUT_CSV = RESULTS_DIR / "negative_energy_geometry_distances.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_centroids_distance_mic(cif_path: str, select_elements: list = None) -> float:
    """
    Reads a CIF file, unwraps/partitions layers dynamically,
    and calculates the Euclidean distance between their geometric centers
    strictly adhering to Minimum Image Convention (MIC).
    """
    try:
        atoms = read(str(cif_path))
        validate_structure(atoms)

        cell = atoms.get_cell().array
        pbc = atoms.get_pbc()
        positions = atoms.get_positions()
        symbols = np.array(atoms.get_chemical_symbols())

        # Wrap positions centered at 0.5
        positions_wrapped = wrap_positions_custom(positions, cell, pbc)
        atoms.set_positions(positions_wrapped)

        # Filter elements if specified
        if select_elements is not None:
            mask = np.isin(symbols, select_elements)
            positions_wrapped = positions_wrapped[mask]
            symbols = symbols[mask]
            if len(positions_wrapped) == 0:
                return None
            atoms = atoms[mask]

        layers = partition_layers_dynamically(atoms, L=2)
        c_lower = calculate_layer_centroids(layers[0])
        c_upper = calculate_layer_centroids(layers[1])

        # Compute pairwise MIC displacement
        _, dist = min_image_displacement(c_lower[None, :], c_upper[None, :], cell, pbc)
        return float(dist[0, 0])

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

    # Standardize paths to prevent path separator mismatches
    df_features['normalized_path'] = df_features['file_path'].apply(lambda p: os.path.normpath(p).lower())
    df_disp['normalized_path'] = df_disp['file_path'].apply(lambda p: os.path.normpath(p).lower())

    df_merged = pd.merge(df_features, df_disp, on='normalized_path', suffixes=('_feat', '_disp'))

    # Filter for structures with negative energies (adjusted_energy < 0)
    df_negative = df_merged[df_merged['adjusted_energy'] < 0].copy()
    print(f"Found {len(df_negative)} structures with negative energy (adjusted_energy < 0).")

    results = []

    for idx, row in tqdm(df_negative.iterrows(), total=len(df_negative), desc="Calculating geometry center distances"):
        cif_path = row['file_path_feat']

        all_atoms_dist = calculate_centroids_distance_mic(cif_path, select_elements=None)
        carbon_dist = calculate_centroids_distance_mic(cif_path, select_elements=['C'])

        X = row['X']
        Y = row['Y']
        Z = row['Z']
        displacement_dist = float(np.sqrt(X**2 + Y**2 + Z**2))

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
    print(df_results.head(5))


if __name__ == "__main__":
    main()
