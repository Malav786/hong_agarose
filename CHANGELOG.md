# Changelog

All notable changes to the NanoStack 3D codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-16

### Added
- **Security Hardening**:
  - Implemented `_validate_safe_cif_path` containment check with directory and `.cif` extension whitelisting on `/api/view_cif` and `/api/download_cif`.
  - Added public demonstration read-only guard (`NANOSTACK_READ_ONLY`) and administrative authentication (`X-Admin-Key`) on `/api/run_pipeline`.
  - Added single-job concurrency locking (`_PIPELINE_LOCK`) to prevent CPU/memory exhaustion.
  - Added `/api/health` system monitoring endpoint.
- **Scientific Correctness & Consistency**:
  - Standardized inference on canonical ONNX engine (`StackingPredictor`) with Joblib fallback, reproducing the exact 300 stable bilayers in `negative_2_cifs`.
  - Fixed multilayer candidate clash detection by dynamically computing threshold matrices per layer template independently of input atom ordering.
  - Added atomic CIF writing (`atomic_write_cif`) to protect against file corruption under multithreaded generation.
- **Packaging & Portability**:
  - Added standard `pyproject.toml` configuration with optional dependencies (`web`, `dev`, `ml-extra`).
  - Added `nanostack3d` CLI tool (`verify`, `serve`).
  - Added deterministic lockfile `requirements-lock.txt`.
- **Automated Testing**:
  - Added `script/tests/test_security.py` (8 automated test cases).
  - Added `script/tests/test_inference_parity.py` (3 parity and benchmark validation cases).
  - Added `script/tests/test_generator_robustness.py` (atom-ordering and fallback tests).
- **Licensing & Documentation**:
  - Added Research Attribution and Academic Use License (`LICENSE`).
  - Added machine-readable `CITATION.cff` and `NOTICE`.
  - Added `CONTRIBUTING.md`, `SECURITY.md`, `.editorconfig`.
  - Added GitHub Actions workflows (`ci.yml`, `security.yml`, `reproducibility.yml`).

### Changed
- Removed hardcoded personal email addresses from `amarel_run.slurm`.
- Removed destructive `shutil.rmtree()` on curated `negative_*_cifs` folders during pipeline execution.

### Preserved
- 100% cryptographic SHA-256 preservation of all original exploratory notebooks in `script/*.ipynb`.
- All original research datasets, checkpoints, CIF structures, and reported baseline figures.
