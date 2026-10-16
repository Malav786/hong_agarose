# NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor

[![CI](https://github.com/Malav786/hong_agarose/actions/workflows/ci.yml/badge.svg)](https://github.com/Malav786/hong_agarose/actions/workflows/ci.yml)
[![Security Scan](https://github.com/Malav786/hong_agarose/actions/workflows/security.yml/badge.svg)](https://github.com/Malav786/hong_agarose/actions/workflows/security.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Research%20Attribution-green.svg)](LICENSE)

An automated computational chemistry and machine learning framework for recursive generation, geometric interface feature extraction, and thermodynamic stability prediction of multi-layer ($N$-layer) molecular crystal configurations.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Scientific Problem & Methodology](#2-scientific-problem--methodology)
3. [Key Features](#3-key-features)
4. [Scientific Limitations & Assumptions](#4-scientific-limitations--assumptions)
5. [Research Results & Evaluation Protocols](#5-research-results--evaluation-protocols)
6. [Cross-Platform Installation](#6-cross-platform-installation)
7. [Quick-Start Guide](#7-quick-start-guide)
8. [CLI Usage](#8-cli-usage)
9. [Web Application Usage](#9-web-application-usage)
10. [Model & Dataset Requirements](#10-model--dataset-requirements)
11. [Hardware Requirements (CPU / GPU)](#11-hardware-requirements-cpu--gpu)
12. [API Documentation & Security Controls](#12-api-documentation--security-controls)
13. [Reproducing Published Results](#13-reproducing-published-results)
14. [Tests & Verification](#14-tests--verification)
15. [Deployment & Containerization](#15-deployment--containerization)
16. [License & Citation Requirements](#16-license--citation-requirements)
17. [Known Limitations & Future Work](#17-known-limitations--future-work)
18. [Acknowledgments & Third-Party Notices](#18-acknowledgments--third-party-notices)

---

## 1. Project Overview

Predicting stable multi-layer stacking configurations of 2D and pseudo-2D molecular materials (e.g., agarose fibrils, polysaccharide chains, organic molecular sheets) requires exploring extensive rotational ($r$) and translational ($t$) conformational landscapes. Full quantum-chemical Density Functional Theory (DFT) calculations on hundreds of thousands of candidate geometries are computationally intractable.

**NanoStack 3D** solves this combinatorial bottleneck by combining:
* Numba JIT-accelerated periodic boundary clash detection with minimum image convention (MIC).
* Compact interfacial contact descriptors capturing physical hydrogen bonding and steric clashes.
* A calibrated stacking ensemble classifier (Random Forest, XGBoost, LightGBM, MLP) with Platt scaling.
* A multi-interface consensus filtering rule for recursive $N$-layer growth.

---

## 2. Scientific Problem & Methodology

### Recursive Multi-Layer Pipeline

```
[Curated Stable (L-1) Parents]
              │
              ▼
 1. Monolayer Stacking (Upper & Lower Sides)
    ├─ Bisection gap alignment under PBC
    ├─ Dynamic element threshold clash filtering (JIT)
    └─ Vacuum cell enlargement (50 Å buffer)
              │
              ▼
 2. Multi-Interface Feature Extraction
    └─ 8 Interatomic contact descriptors per interface
              │
              ▼
 3. Calibrated ML Consensus Classification
    ├─ Canonical ONNX inference per interface
    └─ Consensus rule: Stable iff ALL (L-1) interfaces are stable
              │
              ▼
 [Curated Stable L-Layer Structures]
```

### Physical Descriptors
The model relies on 8 physical interface descriptors derived from pairwise interatomic distances under periodic boundary conditions within defined van der Waals cutoffs:
* `avg_OO_dist`, `count_OO`: O...O hydrogen bond network spacing (cutoff: 3.16 Å).
* `avg_HO_dist`, `count_HO`: H...O hydrogen bonding contacts (cutoff: 2.83 Å).
* `avg_OH_dist`, `count_OH`: Complementary O...H contacts (cutoff: 2.83 Å).
* `avg_HH_dist`, `count_HH`: H...H steric clash repulsion (cutoff: 2.50 Å).

---

## 3. Key Features

* **High Performance**: Numba JIT-compiled math operations with multi-core thread parallelism.
* **Canonical ONNX Inference**: Sub-millisecond single-sample inference (0.53 ms p50) with automated Joblib fallback.
* **Atomic File Writes**: Thread-safe temporary-file atomic renaming prevents half-written or corrupted CIFs under heavy concurrency.
* **Interactive 3D Web Dashboard**: Integrated 3Dmol.js viewer, real-time telemetry, and non-blocking streaming execution.
* **Strict Security Hardening**: Path containment allowlisting, public read-only demonstration mode, and administrative execution guards.
* **Cross-Platform Compatibility**: Tested and verified across Windows, Linux, and macOS.

---

## 4. Scientific Limitations & Assumptions

1. **Rigid-Monomer Assumption**: Candidate generation treats each 78-atom monolayer as internally rigid, sampling rigid-body translations and rotations before stacking.
2. **Interpolation vs. Out-of-Distribution Rotations**: While the model achieves an MCC of $0.8421$ on standard stratified random test splits (interpolating within known rotation angles $r0 \dots r340$), evaluating performance on entirely unseen rotation families via 5-fold Stratified Group Cross-Validation yields MCC $0.7125 \pm 0.1174$.
3. **Consensus Heuristic**: Multi-layer stability ($\Delta E \le 0$) is evaluated via pairwise adjacent interface stability. Non-adjacent third-body van der Waals screening is approximated through the calibrated decision threshold.

---

## 5. Research Results & Evaluation Protocols

| Evaluation Protocol | Metric | Baseline Value | Standard Error / 95% CI |
| ------------------- | ------ | -------------- | ----------------------- |
| **Standard Random Split (80/20)** | MCC | **0.8421** | $[0.7706, 0.9059]$ |
| Standard Random Split (80/20) | ROC-AUC | **0.9789** | - |
| Standard Random Split (80/20) | Accuracy | **96.92%** | - |
| Standard Random Split (80/20) | Brier Score Loss | **0.0245** | - |
| Standard Random Split (80/20) | Expected Calibration Error | **0.0272** | - |
| **Rotation Group-Aware CV (5-Fold)** | Group MCC | **0.7125** | $\pm 0.1174$ |
| **Curated Stable Bilayers ($L=2$)** | Candidate Count | **300** | Exact disk match |
| **Predicted Stable Multilayers ($L=3$)** | Consensus Stable | **155** | Exact disk match |
| **Predicted Stable Multilayers ($L=4$)** | Consensus Stable | **214** | Exact disk match |
| **Predicted Stable Multilayers ($L=5$)** | Consensus Stable | **228** | Exact disk match |

---

## 6. Cross-Platform Installation

### Supported Environments
* **Python**: 3.10, 3.11, 3.12 (Python 3.13 supported with pre-built wheels).
* **Operating Systems**: Windows 10/11, Ubuntu 20.04+, macOS (Apple Silicon / Intel).

### Step 1: Clone Repository
```bash
git clone https://github.com/Malav786/hong_agarose.git
cd hong_agarose
```

### Step 2: Create Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Package
```bash
# Core library and CLI:
pip install -e .

# With Web Dashboard and Development Tools:
pip install -e ".[web,dev]"
```

For exact package version locking matching our reproduction environment, use:
```bash
pip install -r requirements-lock.txt
```

---

## 7. Quick-Start Guide

Verify your installation:
```bash
nanostack3d verify
```

Expected output:
```
==================================================
NanoStack 3D System Verification (v1.0.0)
==================================================
Python Platform : 3.11.x (or 3.12 / 3.13)
Project Root    : ...
Threshold Config: EXISTS
ONNX Model      : EXISTS
Joblib Model    : EXISTS
Predictor Engine: ACTIVE (ONNX Runtime)
Decision Thresh : 0.5368
Active Features : 8 features configured
[STATUS] System is READY for simulation and prediction.
```

---

## 8. CLI Usage

The package installs the `nanostack3d` executable:

* **System Status & Model Check**:
  ```bash
  nanostack3d verify
  ```

* **Launch Web Dashboard**:
  ```bash
  nanostack3d serve --host 127.0.0.1 --port 8000
  ```

---

## 9. Web Application Usage

1. Launch the server:
   ```bash
   python final_system/main.py
   ```
   Or via Uvicorn:
   ```bash
   uvicorn final_system.main:app --host 127.0.0.1 --port 8000
   ```

2. Open `http://127.0.0.1:8000` in your web browser.
3. Explore the **3D Molecular Viewer**: select any stable configuration from the curated bilayer or multilayer libraries to render with 3Dmol.js.
4. Inspect model metrics, Platt calibration curves, and feature importance rankings.

---

## 10. Model & Dataset Requirements

* `data/`: Contains raw bilayer CIF configurations across rotation families ($r0 \dots r340$) and displacement series ($t0 \dots t9.6$), along with `file_energy.csv`.
* `results/model_threshold.json`: Model configuration containing feature names and optimal decision threshold ($0.5368$).
* `results/pipeline_model_calibrated.onnx`: Canonical exported ONNX stacking classifier.
* `results/pipeline_model_calibrated.joblib`: Secondary Scikit-learn serialized pipeline.

---

## 11. Hardware Requirements (CPU / GPU)

* **CPU**: Multi-core x86-64 or ARM64 processor (4+ cores recommended for parallel candidate generation).
* **RAM**: Minimum 4 GB RAM (8 GB+ recommended for generating large multi-layer libraries).
* **GPU**: Optional. ONNX Runtime and Numba automatically run on CPU using optimized vector instructions. No CUDA GPU is required.

---

## 12. API Documentation & Security Controls

### Endpoints

| Endpoint | Method | Description | Security Controls |
| -------- | ------ | ----------- | ----------------- |
| `/api/health` | GET | Health & readiness status | Public |
| `/api/status` | GET | Workspace statistics and CIF index | In-memory cached (5s TTL) |
| `/api/view_cif` | GET | Stream raw CIF file for 3D viewer | Strict path containment; `.cif` only |
| `/api/download_cif` | GET | Download single CIF file | Strict path containment; `.cif` only |
| `/api/download_zip` | GET | Download ZIP bundle of stable CIFs | Validated layer index $2 \le L \le 20$ |
| `/api/metrics` | GET | Model evaluation metrics & features | Public |
| `/api/run_pipeline` | POST | Trigger $N$-layer generation pipeline | Read-only demo guard; `X-Admin-Key` required; concurrency lock |

### Security Environment Variables
* `NANOSTACK_READ_ONLY`: Set to `"true"` (default) to disable pipeline generation on public demo servers.
* `NANOSTACK_ADMIN_KEY`: Administrative authorization token required to execute `POST /api/run_pipeline`.

---

## 13. Reproducing Published Results

To reproduce model metrics and feature extraction:

```bash
# Run automated reproduction verification
pytest script/tests/test_inference_parity.py -v
```

To run group-aware cross-validation across rotation families:
```bash
python -c "
from script.core.models import evaluate_group_cv
import pandas as pd
df = pd.read_csv('results/features_original_model_ready.csv')
scores = evaluate_group_cv(df)
print(scores)
"
```

---

## 14. Tests & Verification

Execute the complete automated test suite:

```bash
pytest script/tests/ -v
```

All 18 tests cover:
1. `test_pipeline_integrity.py`: Conservation of atoms, MIC PBC invariance, metric benchmarks.
2. `test_security.py`: Path traversal blocking, admin authentication, data immutability.
3. `test_inference_parity.py`: Exact 300-structure reproduction with ONNX.
4. `test_generator_robustness.py`: Atom-ordering invariance and vdW fallback.

---

## 15. Deployment & Containerization

### Docker Deployment
Build and run using Docker:

```bash
docker build -t nanostack3d .
docker run -p 8000:8000 -e NANOSTACK_READ_ONLY=true nanostack3d
```

### HPC Deployment (Slurm)
An Amarel/Slurm launch script is provided in `amarel_run.slurm`. Edit the email placeholder before submitting:
```bash
sbatch amarel_run.slurm
```

---

## 16. License & Citation Requirements

This software and its models are distributed under the **NanoStack 3D Research Attribution and Academic Use License** (see [LICENSE](LICENSE)).

### Mandatory Citation
If you use NanoStack 3D, its models, or generated multilayer candidate structures in scientific work, you must cite:

```bibtex
@article{nanostack3d2026,
  title={[PAPER TITLE, DOI, OR OFFICIAL CITATION]},
  author={[AUTHOR'S FULL LEGAL OR PREFERRED PUBLICATION NAME]},
  journal={[JOURNAL NAME OR PREPRINT REPOSITORY]},
  year={2026},
  doi={[PAPER DOI PLACEHOLDER]}
}

@software{nanostack3d_code,
  title={NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor},
  author={[AUTHOR'S FULL LEGAL OR PREFERRED PUBLICATION NAME]},
  year={2026},
  url={https://github.com/Malav786/hong_agarose}
}
```

(Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff)).

---

## 17. Known Limitations & Future Work

* **Solvent & Thermal Effects**: Geometries and energies are evaluated at 0 K in vacuum. Future releases will integrate explicit solvent and finite-temperature molecular dynamics screening.
* **Expanded Element Geometries**: Currently optimized for polysaccharide systems ($C, H, O$). Generalization to heteroatom inorganic sheets ($MoS_2, WS_2, hBN$) is in progress.

---

## 18. Acknowledgments & Third-Party Notices

We gratefully acknowledge the developers and open-source communities of:
* Atomic Simulation Environment (ASE)
* Scikit-Learn, XGBoost, and LightGBM
* ONNX Runtime
* FastAPI, Uvicorn, and 3Dmol.js

See [NOTICE](NOTICE) for third-party license details.
