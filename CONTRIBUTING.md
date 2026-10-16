# Contributing to NanoStack 3D

Thank you for your interest in contributing to **NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor**.

We welcome contributions from computational chemists, materials scientists, and scientific software engineers.

---

## 📜 Code of Conduct & Academic Integrity

As a research-driven project:
1. **Provenance & Reproducibility**: Contributions must preserve physical units, coordinate frame conventions, and scientific reproducibility.
2. **Notebook Preservation Policy**: Never mutate original exploratory notebooks in `script/*.ipynb`. Any new reusable functionality should be contributed as tested modules in `script/core/`.
3. **No Secret or Credential Exposure**: Never commit API keys, personal email addresses, private file paths, or credentials.

---

## 🛠️ Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Malav786/hong_agarose.git
   cd hong_agarose
   ```

2. **Create an isolated Python environment** (Python 3.10 to 3.12 recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,web]"
   ```

4. **Verify System Integrity**:
   ```bash
   nanostack3d verify
   ```

---

## 🧪 Testing Guidelines

Before opening a pull request, all automated test suites must pass:

```bash
pytest script/tests/ -v
```

The test suite covers:
* `test_pipeline_integrity.py`: Conservation of atoms, Minimum Image Convention (MIC) PBC translation invariance, and calibrated stacking MCC benchmarks.
* `test_security.py`: Path traversal protection, file allowlisting, and read-only demo mode enforcement.
* `test_inference_parity.py`: Exact 300-structure reproduction with canonical ONNX runtime and Joblib parity checks.
* `test_generator_robustness.py`: Atom-ordering invariance and periodic boundary clash checks.

---

## 🔀 Submitting a Pull Request

1. Create a feature branch from `master` (`git checkout -b feature/your-feature-name`).
2. Make minimal, focused, well-documented changes.
3. Ensure no trailing whitespace or CRLF issues exist.
4. Run the test suite and confirm 100% pass rate.
5. Submit a pull request with a clear description of the scientific rationale and test verification results.
