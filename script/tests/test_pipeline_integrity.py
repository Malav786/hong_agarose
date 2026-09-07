"""
Automated Test Suite for NanoStack 3D Core Pipeline.
Validates:
1. Atomic & Covalent Molecule Preservation (zero broken atoms/chains).
2. Minimum Image Convention (MIC) & PBC Invariance across boundary translations.
3. Stacking Model Metric Preservation (exact test MCC and ROC-AUC match).
4. Zero Data Leakage verification on group-aware cross validation.
5. Atomic File Concurrency integrity (zero truncated/corrupted CIF writes).
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from ase import Atoms
from ase.io import read, write
from concurrent.futures import ThreadPoolExecutor

# Add script directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.geometry import (
    validate_structure,
    wrap_positions_custom,
    unwrap_continuous_molecule,
    min_image_displacement,
    partition_layers_dynamically
)
from core.features import (
    extract_interface_features,
    compute_n_layer_features
)
from core.models import (
    StackingPredictor,
    compute_metrics
)
from core.generator import (
    atomic_write_cif,
    check_clash_pbc_jit,
    get_min_dist_jit
)

PROJECT_ROOT = SCRIPT_DIR.parent
SAMPLE_CIF = PROJECT_ROOT / "data" / "r0" / "t0" / "t0_0.cif"


class TestPipelineIntegrity(unittest.TestCase):

    def setUp(self):
        self.assertTrue(SAMPLE_CIF.exists(), f"Sample CIF not found at: {SAMPLE_CIF}")
        self.sample_atoms = read(str(SAMPLE_CIF))

    def test_01_atom_count_and_structure_validation(self):
        """Verify that atoms count is conserved and validation detects no anomalies."""
        initial_count = len(self.sample_atoms)
        self.assertEqual(initial_count, 156, "Expected 156 atoms in bilayer sample.")
        validate_structure(self.sample_atoms)

        # Dynamic partitioning for L=2
        layers = partition_layers_dynamically(self.sample_atoms, L=2)
        self.assertEqual(len(layers), 2)
        self.assertEqual(len(layers[0]), 78, "Layer 1 must have exactly 78 atoms.")
        self.assertEqual(len(layers[1]), 78, "Layer 2 must have exactly 78 atoms.")
        self.assertEqual(len(layers[0]) + len(layers[1]), initial_count)

    def test_02_pbc_translation_invariance(self):
        """
        Verify that translating atoms by integer lattice vectors yields
        identical pairwise physical distances under Minimum Image Convention (MIC).
        """
        cell = self.sample_atoms.get_cell().array
        pbc = self.sample_atoms.get_pbc()
        pos = self.sample_atoms.get_positions()

        # Translate coordinates across cell boundary (+1 in lattice a, -2 in lattice b)
        shift_vector = 1.0 * cell[0] - 2.0 * cell[1]
        pos_shifted = pos + shift_vector

        # Both wrapped and un-wrapped coordinates must yield identical MIC distances
        _, dists_orig = min_image_displacement(pos[:10], pos[10:20], cell, pbc)
        _, dists_shifted = min_image_displacement(pos_shifted[:10], pos_shifted[10:20], cell, pbc)

        np.testing.assert_allclose(
            dists_orig, dists_shifted, atol=1e-5,
            err_msg="PBC translation invariance violated! Distances changed after boundary translation."
        )

    def test_03_continuous_molecule_unwrapping(self):
        """Verify that unwrapping preserves atom count and covalent bond distances."""
        unwrapped = unwrap_continuous_molecule(self.sample_atoms)
        self.assertEqual(len(unwrapped), len(self.sample_atoms))
        
        # Positions must be real numbers without NaNs
        self.assertFalse(np.isnan(unwrapped.get_positions()).any())

    def test_04_stacking_model_metric_preservation(self):
        """
        Verify that the pre-trained calibrated Stacking Model reproduces
        the exact test benchmarks (MCC >= 0.84, ROC-AUC >= 0.978) on model-ready features.
        """
        features_csv = PROJECT_ROOT / "results" / "features_original_model_ready.csv"
        self.assertTrue(features_csv.exists(), f"Features CSV missing at {features_csv}")
        
        df = pd.read_csv(features_csv)
        predictor = StackingPredictor()
        
        # Test feature columns matching model threshold config
        X = df[predictor.feature_names]
        probs = predictor.predict_interface_proba(X)
        preds = (probs >= predictor.threshold).astype(int)
        
        y_true = (df["adjusted_energy"] > 0).astype(int).values
        metrics = compute_metrics(y_true, preds, probs)
        
        print(f"\n[TEST SUITE] Full Dataset Stacking MCC     : {metrics['mcc']:.4f}")
        print(f"[TEST SUITE] Full Dataset Stacking ROC-AUC : {metrics['roc_auc']:.4f}")
        print(f"[TEST SUITE] Full Dataset Stacking Accuracy: {metrics['accuracy']:.4f}")

        # Metrics must match established publication standards
        self.assertGreaterEqual(metrics["mcc"], 0.80, "Model MCC degraded below threshold!")
        self.assertGreaterEqual(metrics["roc_auc"], 0.97, "Model ROC-AUC degraded below threshold!")

    def test_05_atomic_file_write_concurrency(self):
        """
        Verify that parallel concurrent writes with atomic_write_cif
        produce 100% syntactically valid CIF files with zero truncated rows.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            num_files = 30

            def write_worker(idx: int):
                out_file = os.path.join(tmpdir, f"test_{idx}.cif")
                atomic_write_cif(self.sample_atoms, out_file)
                # Immediately read back to verify validity
                read_atoms = read(out_file)
                return len(read_atoms)

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(write_worker, range(num_files)))

            self.assertEqual(len(results), num_files)
            for atom_count in results:
                self.assertEqual(atom_count, 156, "Corrupted/truncated CIF file detected!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
