"""
NanoStack 3D Command-Line Interface (CLI).
Provides command-line utilities for:
- Environment and model readiness checks
- Interface contact feature extraction
- Consensus stability prediction
- Web dashboard service launching
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

__version__ = "1.0.0"


def cmd_verify(args: argparse.Namespace) -> int:
    """Verifies environment, dependencies, and model readiness."""
    print("==================================================")
    print(f"NanoStack 3D System Verification (v{__version__})")
    print("==================================================")
    print(f"Python Platform : {sys.version.split()[0]} ({sys.platform})")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = project_root / "results"
    onnx_path = results_dir / "pipeline_model_calibrated.onnx"
    joblib_path = results_dir / "pipeline_model_calibrated.joblib"
    thresh_path = results_dir / "model_threshold.json"

    print(f"Project Root    : {project_root}")
    print(f"Threshold Config: {'EXISTS' if thresh_path.exists() else 'MISSING'}")
    print(f"ONNX Model      : {'EXISTS' if onnx_path.exists() else 'MISSING'}")
    print(f"Joblib Model    : {'EXISTS' if joblib_path.exists() else 'MISSING'}")

    try:
        from .models import StackingPredictor
        predictor = StackingPredictor()
        engine = "ONNX Runtime" if predictor.use_onnx else "Joblib (Scikit-Learn)"
        print(f"Predictor Engine: ACTIVE ({engine})")
        print(f"Decision Thresh : {predictor.threshold:.4f}")
        print(f"Active Features : {len(predictor.feature_names)} features configured")
        print("[STATUS] System is READY for simulation and prediction.")
        return 0
    except Exception as ex:
        print(f"[ERROR] Failed to initialize predictor: {ex}")
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Launches the FastAPI interactive web dashboard."""
    project_root = Path(__file__).resolve().parent.parent.parent
    final_system_dir = project_root / "final_system"
    sys.path.insert(0, str(final_system_dir))

    try:
        import uvicorn
        from final_system.main import app
        print(f"Launching NanoStack 3D Web Service on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except ImportError:
        print("[ERROR] uvicorn and fastapi are required to run the web dashboard.")
        print("Install them via: pip install '.[web]'")
        return 1


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    """Runs batch multi-layer generation, extraction, and prediction on HPC or workstation."""
    project_root = Path(__file__).resolve().parent.parent.parent
    final_system_dir = project_root / "final_system"
    sys.path.insert(0, str(final_system_dir))

    from pipeline import run_n_layer_generation, run_n_layer_extraction, run_n_layer_predictions
    import numpy as np

    L = args.layer
    cores = args.cores if args.cores else min(32, os.cpu_count() or 4)
    input_dir = Path(args.input) if args.input else (project_root / f"negative_{L-1}_cifs")
    out_dir = Path(args.output_dir) if args.output_dir else (project_root / "results" / f"{L}_layer_hpc_run")
    candidates_dir = out_dir / "candidates"
    csv_path = out_dir / f"{L}_layer_features.csv"
    dest_dir = out_dir / "stable_cifs"

    if not input_dir.exists():
        print(f"[ERROR] Parent structure directory missing: {input_dir}")
        return 1

    angles = np.array(
        args.angles if args.angles
        else [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    )

    print("==================================================")
    print(f"NanoStack 3D Batch Pipeline Execution (L={L})")
    print("==================================================")
    print(f"Parent Input    : {input_dir}")
    print(f"Output Workspace: {out_dir}")
    print(f"Worker Cores    : {cores}")
    print(f"Sampling Angles : {len(angles)} rotations")

    # Step 1: Generation
    if args.mode in ("all", "generate"):
        print(f"\n[Stage 1/3] Generating L={L} candidate structures...")
        run_n_layer_generation(
            L=L,
            base_in=str(input_dir),
            out_root=str(candidates_dir),
            chosen_angles=angles,
            cores=cores,
            logger=print
        )

    # Step 2: Extraction
    if args.mode in ("all", "extract"):
        print(f"\n[Stage 2/3] Extracting geometric interface features...")
        run_n_layer_extraction(
            L=L,
            base_dir=str(candidates_dir),
            out_csv=str(csv_path),
            cores=cores,
            logger=print
        )

    # Step 3: Prediction
    if args.mode in ("all", "predict"):
        print(f"\n[Stage 3/3] Predicting consensus thermodynamic stability...")
        model_path = project_root / "results" / "pipeline_model_calibrated.onnx"
        meta_path = project_root / "results" / "model_threshold.json"
        run_n_layer_predictions(
            L=L,
            csv_path=str(csv_path),
            model_path=str(model_path),
            metadata_path=str(meta_path),
            dest_dir=str(dest_dir),
            source_dir=str(candidates_dir),
            logger=print
        )

    print(f"\n[SUCCESS] Pipeline completed successfully. Results saved in: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nanostack3d",
        description="NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor"
    )
    parser.add_argument("--version", action="version", version=f"nanostack3d {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # verify
    p_verify = subparsers.add_parser("verify", help="Check dependencies, models, and system status")
    p_verify.set_defaults(func=cmd_verify)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch interactive web service dashboard")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    p_serve.set_defaults(func=cmd_serve)

    # run-pipeline (Batch HPC / Slurm CLI)
    p_run = subparsers.add_parser("run-pipeline", help="Execute high-throughput N-layer pipeline (Slurm/HPC)")
    p_run.add_argument("--layer", "-l", type=int, required=True, help="Target layer L (e.g. 3, 4, 5)")
    p_run.add_argument("--mode", choices=["generate", "extract", "predict", "all"], default="all", help="Pipeline stages to run")
    p_run.add_argument("--input", "-i", type=str, default=None, help="Input directory of parent (L-1) structures")
    p_run.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for results (default: results/{L}_layer_hpc_run)")
    p_run.add_argument("--cores", "-c", type=int, default=None, help="Number of CPU worker threads (default: autodetect)")
    p_run.add_argument("--angles", nargs="+", type=int, default=None, help="Custom sampling angles in degrees")
    p_run.set_defaults(func=cmd_run_pipeline)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
