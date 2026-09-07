"""
Core Geometry and Periodic Boundary Condition (PBC) Engine.
Ensures rigorous Minimum Image Convention (MIC), continuous molecular chain unwrapping
without bond breakage, and dynamic layer partitioning for arbitrary atom counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from typing import Tuple, List, Optional, Dict, Any

# Standard van der Waals radii (Bondi / Rowland & Taylor)
VDW_RADII: Dict[str, float] = {
    'C': 1.70,
    'H': 1.20,
    'O': 1.52,
    'N': 1.55,
    'S': 1.80
}

DEFAULT_VDW_CUTOFFS: Dict[Tuple[str, str], float] = {
    ('H', 'H'): 2.50,  # 2 * 1.20 + 0.10
    ('O', 'O'): 3.16,  # 2 * 1.53 + 0.10
    ('H', 'O'): 2.83,  # 1.20 + 1.53 + 0.10
    ('O', 'H'): 2.83
}


def validate_structure(atoms: Atoms) -> None:
    """Validates basic physical integrity of an Atoms structure."""
    if atoms is None or len(atoms) == 0:
        raise ValueError("Structure contains zero atoms.")
    if atoms.get_cell().volume <= 0:
        raise ValueError(f"Invalid unit cell volume: {atoms.get_cell().volume}")
    if np.isnan(atoms.get_positions()).any():
        raise ValueError("NaN detected in atomic positions.")


def wrap_positions_custom(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> np.ndarray:
    """
    Wraps Cartesian coordinates into the unit cell centered around `center`.
    Guarantees that all fractional coordinates fall strictly inside [center - 0.5, center + 0.5).
    """
    inv_cell = np.linalg.inv(cell)
    frac = np.dot(positions, inv_cell)
    for i in range(3):
        if pbc[i]:
            shift = frac[:, i] - center[i] + 0.5
            frac[:, i] = (shift % 1.0) + center[i] - 0.5
    return np.dot(frac, cell)


def min_image_displacement(
    pos_source: np.ndarray,
    pos_target: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes pairwise Cartesian displacement vectors and Euclidean distances
    under exact Minimum Image Convention (MIC) across periodic boundaries.
    
    Parameters:
        pos_source: (N, 3) Cartesian coordinates
        pos_target: (M, 3) Cartesian coordinates
        cell: (3, 3) Lattice cell vectors
        pbc: (3,) Boolean periodic flags
        
    Returns:
        diff_cart: (N, M, 3) Minimum-image displacement vectors (target - source)
        dists: (N, M) Pairwise Euclidean distances in Å
    """
    inv_cell = np.linalg.inv(cell)
    frac_src = pos_source @ inv_cell
    frac_tgt = pos_target @ inv_cell
    
    # Pairwise fractional difference: (N, M, 3)
    diff_frac = frac_tgt[None, :, :] - frac_src[:, None, :]
    
    for i in range(3):
        if pbc[i]:
            diff_frac[:, :, i] -= np.round(diff_frac[:, :, i])
            
    diff_cart = diff_frac @ cell
    dists = np.linalg.norm(diff_cart, axis=2)
    return diff_cart, dists


def unwrap_continuous_molecule(
    atoms: Atoms,
    bond_cutoff_factor: float = 1.25
) -> Atoms:
    """
    Unwraps a periodic molecular system so that covalently bonded polymer chains 
    remain fully contiguous and are NEVER severed across periodic cell boundaries.
    
    Uses Breadth-First Search (BFS) starting from the centroid root atom,
    ensuring each neighbor is brought to the nearest periodic image of its bonded partner.
    """
    atoms = atoms.copy()
    cell = atoms.get_cell().array
    pbc = atoms.get_pbc()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    
    if n_atoms <= 1 or not any(pbc):
        return atoms

    # Determine pairwise covalent bonding threshold matrix
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell
    
    # Compute pairwise MIC distances
    diff_frac = frac[None, :, :] - frac[:, None, :]
    for i in range(3):
        if pbc[i]:
            diff_frac[:, :, i] -= np.round(diff_frac[:, :, i])
    pairwise_dist = np.linalg.norm(diff_frac @ cell, axis=2)
    
    # Build covalent adjacency list
    cov_radii = {
        'H': 0.31, 'C': 0.76, 'O': 0.66, 'N': 0.71, 'S': 1.05
    }
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i in range(n_atoms):
        r_i = cov_radii.get(symbols[i], 1.5)
        for j in range(i + 1, n_atoms):
            r_j = cov_radii.get(symbols[j], 1.5)
            thresh = (r_i + r_j) * bond_cutoff_factor
            if pairwise_dist[i, j] <= thresh:
                adj[i].append(j)
                adj[j].append(i)

    # BFS traversal to unwrap connected components
    visited = np.zeros(n_atoms, dtype=bool)
    unwrapped_frac = frac.copy()
    
    for root in range(n_atoms):
        if visited[root]:
            continue
        visited[root] = True
        queue = [root]
        
        while queue:
            curr = queue.pop(0)
            curr_f = unwrapped_frac[curr]
            for nbr in adj[curr]:
                if not visited[nbr]:
                    df = frac[nbr] - curr_f
                    for ax in range(3):
                        if pbc[ax]:
                            df[ax] -= np.round(df[ax])
                    unwrapped_frac[nbr] = curr_f + df
                    visited[nbr] = True
                    queue.append(nbr)

    atoms.set_positions(unwrapped_frac @ cell)
    return atoms


def partition_layers_dynamically(
    atoms: Atoms,
    L: int
) -> List[pd.DataFrame]:
    """
    Dynamically partitions an L-layer structure into L individual monolayer DataFrames.
    Does NOT hardcode 78 or 234 atoms; dynamically calculates expected atoms per layer.
    
    Parameters:
        atoms: ASE Atoms object of the full L-layer stack
        L: Number of layers present in the structure
        
    Returns:
        List of L DataFrames, ordered from bottom (lowest Z) to top (highest Z).
    """
    validate_structure(atoms)
    total_atoms = len(atoms)
    
    if total_atoms % L != 0:
        raise ValueError(
            f"Total atom count ({total_atoms}) is not cleanly divisible by layer count L={L}. "
            f"Cannot evenly partition layers."
        )
    atoms_per_layer = total_atoms // L

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    df.insert(0, "Element", symbols)
    
    # Sort along stacking axis (Z)
    df_sorted = df.sort_values(by="z").reset_index(drop=True)
    
    layers = [
        df_sorted.iloc[i * atoms_per_layer : (i + 1) * atoms_per_layer].copy().reset_index(drop=True)
        for i in range(L)
    ]
    return layers


def calculate_layer_centroids(layer_df: pd.DataFrame) -> np.ndarray:
    """Calculates the geometric centroid of heavy atoms (C and O) in a monolayer."""
    heavy = layer_df[layer_df["Element"].isin(["C", "O"])]
    if not heavy.empty:
        return heavy[["x", "y", "z"]].values.mean(axis=0)
    return layer_df[["x", "y", "z"]].values.mean(axis=0)
