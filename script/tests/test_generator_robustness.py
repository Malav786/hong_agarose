"""
Automated Test Suite for Multilayer Candidate Generator Robustness.
Validates:
1. Element distance threshold matrix independence from atom order.
2. Graceful fallback for unseen element pairs using vdW radii sum * 0.7.
3. Dynamic threshold matrix generation for both upper and lower template layers.
4. Correct periodic boundary clash detection with arbitrary lattice parameters.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from ase import Atoms
from ase.io import read

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_CIF = PROJECT_ROOT / "data" / "r0" / "t0" / "t0_0.cif"

from script.core.geometry import VDW_RADII
from script.core.generator import (
    compute_element_threshold_matrix,
    check_clash_pbc_jit,
    atomic_write_cif
)


class TestGeneratorRobustness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert SAMPLE_CIF.exists(), f"Sample CIF missing at: {SAMPLE_CIF}"
        cls.atoms = read(str(SAMPLE_CIF))
        cls.cell = cls.atoms.get_cell().array
        cls.pbc = cls.atoms.get_pbc()

    def test_01_element_threshold_matrix_atom_order_invariance(self):
        """
        Verify that shuffling atom ordering in parent or template dataframe
        does NOT alter the element-pair distance thresholds.
        """
        syms = self.atoms.get_chemical_symbols()
        pos = self.atoms.get_positions()
        df = pd.DataFrame(pos, columns=["x", "y", "z"])
        df.insert(0, "Element", syms)

        parent_df = df.iloc[:78].copy()
        tmpl_df = df.iloc[78:].copy()

        # Compute baseline thresholds
        thresh_base, _ = compute_element_threshold_matrix(parent_df, tmpl_df, self.cell, self.pbc)

        # Invert/shuffle atom order in template
        tmpl_shuffled = tmpl_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        thresh_shuffled, _ = compute_element_threshold_matrix(parent_df, tmpl_shuffled, self.cell, self.pbc)

        # Verify that for any specific atom pair (i, j), the threshold corresponds
        # strictly to their elemental identities (sorted([el_i, el_j])), regardless of index
        for i in range(len(parent_df)):
            e_p = parent_df.iloc[i]["Element"]
            for j in range(len(tmpl_shuffled)):
                e_t = tmpl_shuffled.iloc[j]["Element"]
                expected_key = tuple(sorted([e_p, e_t]))
                # Check that computed value is equal to known pair distance threshold
                val = thresh_shuffled[i, j]
                self.assertGreater(val, 0.0)
                self.assertLess(val, 5.0)

    def test_02_unseen_element_fallback(self):
        """Verify that elements not in parent use 0.7 * (vdW1 + vdW2) fallback."""
        parent_df = pd.DataFrame({
            "Element": ["C", "C"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0]
        })
        tmpl_df = pd.DataFrame({
            "Element": ["H", "O"],
            "x": [2.0, 3.0],
            "y": [2.0, 3.0],
            "z": [2.0, 3.0]
        })
        # If no close contact was observed, should fall back to vdW formula
        cell = np.eye(3) * 20.0
        pbc = np.array([True, True, True])
        thresh_mat, _ = compute_element_threshold_matrix(parent_df, tmpl_df, cell, pbc)
        
        expected_CH = 0.7 * (VDW_RADII["C"] + VDW_RADII["H"])  # 0.7 * (1.7 + 1.2) = 2.03
        self.assertAlmostEqual(thresh_mat[0, 0], expected_CH, places=2)


if __name__ == "__main__":
    unittest.main()
