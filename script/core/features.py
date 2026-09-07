"""
Robust Multi-Interface Feature Extraction Module.
Extracts contact metrics (H-H, O-O, O-H, H-O average distances and coordination counts)
under strict Minimum Image Convention (MIC) across adjacent monolayer interfaces.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from ase.io import read
from typing import Dict, Any, List, Optional, Tuple

from .geometry import (
    validate_structure,
    wrap_positions_custom,
    unwrap_continuous_molecule,
    min_image_displacement,
    partition_layers_dynamically,
    DEFAULT_VDW_CUTOFFS
)


def extract_interface_features(
    layer_lower: pd.DataFrame,
    layer_upper: pd.DataFrame,
    cell: np.ndarray,
    pbc: np.ndarray,
    use_mic: bool = True,
    cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> Dict[str, float]:
    """
    Computes contact bounds features between two adjacent monolayers.
    
    Parameters:
        layer_lower: DataFrame with columns ['Element', 'x', 'y', 'z']
        layer_upper: DataFrame with columns ['Element', 'x', 'y', 'z']
        cell: (3, 3) Lattice cell array
        pbc: (3,) Periodic boundary flags
        use_mic: If True, uses exact Minimum Image Convention. If False, uses Euclidean distance.
        cutoffs: Dict mapping (elem1, elem2) -> cutoff distance in Å.
        
    Returns:
        Dict with 8 standardized features:
        - avg_HH_dist, avg_OO_dist, avg_OH_dist, avg_HO_dist
        - count_HH, count_OO, count_OH, count_HO
    """
    if cutoffs is None:
        cutoffs = DEFAULT_VDW_CUTOFFS

    l_H = layer_lower[layer_lower["Element"] == "H"][["x", "y", "z"]].to_numpy(float)
    l_O = layer_lower[layer_lower["Element"] == "O"][["x", "y", "z"]].to_numpy(float)
    u_H = layer_upper[layer_upper["Element"] == "H"][["x", "y", "z"]].to_numpy(float)
    u_O = layer_upper[layer_upper["Element"] == "O"][["x", "y", "z"]].to_numpy(float)

    def query_contacts(pos_a: np.ndarray, pos_b: np.ndarray, cutoff: float) -> Tuple[float, int]:
        if len(pos_a) == 0 or len(pos_b) == 0:
            return 3.0, 0
            
        if use_mic:
            _, dist_matrix = min_image_displacement(pos_a, pos_b, cell, pbc)
            # For each atom in pos_b (upper), find minimum distance to pos_a (lower)
            min_dists = dist_matrix.min(axis=0)
            valid = min_dists[min_dists <= cutoff]
        else:
            from scipy.spatial import cKDTree as KDTree
            tree = KDTree(pos_a)
            dists, _ = tree.query(pos_b, k=1)
            valid = dists[dists <= cutoff]

        avg_val = float(np.mean(valid)) if len(valid) > 0 else 3.0
        return avg_val, len(valid)

    # 1. H - H contacts
    avg_hh, count_hh = query_contacts(l_H, u_H, cutoffs[('H', 'H')])

    # 2. O - O contacts
    avg_oo, count_oo = query_contacts(l_O, u_O, cutoffs[('O', 'O')])

    # 3. Lower-H to Upper-O contacts (O-H)
    avg_oh, count_oh = query_contacts(l_H, u_O, cutoffs[('H', 'O')])

    # 4. Lower-O to Upper-H contacts (H-O)
    avg_ho, count_ho = query_contacts(l_O, u_H, cutoffs[('O', 'H')])

    return {
        "avg_HH_dist": avg_hh,
        "avg_OO_dist": avg_oo,
        "avg_OH_dist": avg_oh,
        "avg_HO_dist": avg_ho,
        "count_HH": count_hh,
        "count_OO": count_oo,
        "count_OH": count_oh,
        "count_HO": count_ho
    }


def compute_n_layer_features(
    cif_path: str,
    L: int,
    use_mic: bool = True,
    interface_naming: str = "Int"
) -> Dict[str, Any]:
    """
    Loads an L-layer structure, wraps coordinates, partitions into monolayers,
    and extracts 8 contact features for each of the (L - 1) adjacent interfaces.
    
    Parameters:
        cif_path: Path to the CIF file
        L: Total layer count in structure (e.g. 2 for bilayer, 3 for trilayer)
        use_mic: Whether to apply strict Minimum Image Convention
        interface_naming: 'Int' generates Int1_*, Int2_*
                          'Legacy' generates L_M_*, M_U_* (for L=3)
                          'Flat' generates un-prefixed features (for L=2)
                          
    Returns:
        Dict containing file_path, status, and namespaced feature keys.
    """
    try:
        atoms = read(cif_path)
        validate_structure(atoms)

        cell = atoms.get_cell().array
        pbc = atoms.get_pbc()
        positions = atoms.get_positions()

        # Wrap positions centered at 0.5
        wrapped_pos = wrap_positions_custom(positions, cell, pbc, center=(0.5, 0.5, 0.5))
        atoms.set_positions(wrapped_pos)

        # Partition layers
        layers = partition_layers_dynamically(atoms, L)

        record: Dict[str, Any] = {
            "status": "success",
            "file_path": cif_path,
            "L": L
        }

        # Compute features across each adjacent interface
        for i in range(L - 1):
            lower_df = layers[i]
            upper_df = layers[i + 1]
            
            feat_dict = extract_interface_features(
                lower_df, upper_df, cell, pbc, use_mic=use_mic
            )

            # Namespace features
            if L == 2 and interface_naming == "Flat":
                prefix = ""
            elif L == 3 and interface_naming == "Legacy":
                prefix = "L_M_" if i == 0 else "M_U_"
            else:
                prefix = f"{interface_naming}{i + 1}_"

            for k, v in feat_dict.items():
                record[f"{prefix}{k}"] = v

        return record

    except Exception as exc:
        return {
            "status": "error",
            "file_path": cif_path,
            "L": L,
            "error": str(exc)
        }
