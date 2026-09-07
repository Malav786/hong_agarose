# NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor

An automated computational chemistry and machine learning framework for recursive generation, geometric interface feature extraction, and thermodynamic stability prediction of multi-layer ($N$-layer) molecular crystal configurations.

---

## 🔬 Scientific Overview

Predicting stable multi-layer stacking configurations of 2D/pseudo-2D periodic molecular chains (e.g. agarose, cellulose, polysaccharide nanofibrils) requires navigating vast rotational ($r$) and translational ($t$) degrees of freedom. 

Calculating binding energies via full quantum chemical Density Functional Theory (DFT) across thousands of configurations is computationally prohibitive. **NanoStack 3D** establishes a recursive machine-learning workflow:

1. **Bilayer Baseline**: Uses experimental/DFT bilayer binding energies to learn structural contact signatures.
2. **Interface Feature Extraction**: Computes 8 periodic contact features ($H-H, O-O, O-H, H-O$ average distances and coordination counts) within van der Waals limits.
3. **Calibrated ML Classifier**: Combines Random Forest, XGBoost, LightGBM, and MLP via a Stacking Ensemble with Sigmoid Platt Scaling ($MCC \approx 0.842$, $ROC\text{-}AUC \approx 0.979$).
4. **Recursive N-Layer Generation**: Takes stable $(L-1)$-layer structures, stacks a new monolayer above/below, performs Numba JIT-accelerated clash checks and bisection gap alignment, and extracts interface features.
5. **Consensus Stability Rule**: An $L$-layer structure is classified as stable ($\Delta E \le 0$) if and only if **all $L-1$ interfaces** are classified as stable by the model.

---

## 📂 Repository Structure

```
├── data/                  # Raw bilayer CIF configurations (r0 to r340, t0 to t9.6) & file_energy.csv
├── final_system/          # Production application
│   ├── main.py            # FastAPI backend server with streaming log execution
│   ├── pipeline.py        # Generalized N-layer generation, extraction & inference engine
│   └── static/            # Interactive Web UI (HTML5, Vanilla CSS, 3Dmol.js molecular viewer)
├── negative_2_cifs/       # Curated stable bilayer structures (ΔE ≤ 0)
├── negative_3_cifs/       # Predicted stable 3-layer (trilayer) structures
├── negative_4_cifs/       # Predicted stable 4-layer structures
├── negative_5_cifs/       # Predicted stable 5-layer structures
├── results/               # Extracted feature CSVs, model weights (.joblib, .onnx), and threshold metadata
├── reviews/               # In-depth validation reports, feature importances, and group CV analyses
├── script/                # Scientific exploration & development notebooks:
│   ├── TM_gen.ipynb       # 1D K-Means displacement vector calculations
│   ├── calculate_negative_energy_distances.py # Equilibrium centroid distance calculations
│   ├── data_with_model.ipynb # COM alignment, layer splitting, feature extraction, and ML training
│   ├── second_layer.ipynb # Bilayer synthesis & rotation testing
│   └── third_layer.ipynb  # Trilayer candidate generation & multi-interface prediction
└── .gitignore             # Git ignore file for Python, environments, and caches
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Ensure you have Python 3.9+ installed.

```bash
git clone https://github.com/Malav786/hong_agarose.git
cd hong_agarose

# Optional: Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install ase numpy pandas scipy scikit-learn xgboost lightgbm joblib onnxruntime fastapi uvicorn numba tqdm
```

### 2. Running the Interactive Web Dashboard

To launch the web interface with the real-time 3D molecular viewer and pipeline controller:

```bash
cd final_system
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at `http://127.0.0.1:8000` to interactively view stable CIF structures and launch $N$-layer generation jobs.

---

## 📊 Model Performance Highlights

* **Matthews Correlation Coefficient (MCC)**: `0.8421` (95% CI: `[0.7706, 0.9059]`)
* **ROC-AUC**: `0.9789` | **PR-AUC**: `0.9969`
* **Test Accuracy**: `96.92%`
* **Brier Score Loss**: `0.0245` | **Expected Calibration Error (ECE)**: `0.0272`
* **Optimal Decision Threshold**: `0.5368` (Platt calibrated)
