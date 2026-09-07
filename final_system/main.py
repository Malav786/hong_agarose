from __future__ import annotations

import os
import re
import asyncio
import sys
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add current directory to path to import pipeline module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_n_layer_generation, run_n_layer_extraction, run_n_layer_predictions

app = FastAPI(title="NanoStack 3D - Generalized N-Layer Architect")

# Resolve project paths relative to project root
FINAL_SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FINAL_SYSTEM_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
VDW_CONFIG_PATH = RESULTS_DIR / "model_threshold.json"
MODEL_PATH = RESULTS_DIR / "pipeline_model_calibrated.joblib"

# Mount Static UI Files (served from static/ folder)
app.mount("/static", StaticFiles(directory=str(FINAL_SYSTEM_DIR / "static")), name="static")

@app.get("/")
def read_root():
    # Redirect root to index.html
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

# ================= WORKSPACE SYNC & STATS ENDPOINT =================
@app.get("/api/status")
def get_status():
    """
    Scans the workspace directory for bilayers, trilayers, and stable structures.
    """
    # 1) Bilayers Count (L=2 parents)
    bilayers_dir = PROJECT_DIR / "negative_2_cifs"
    bilayers_count = 0
    if bilayers_dir.exists():
        for root, _, files in os.walk(bilayers_dir):
            for f in files:
                if f.lower().endswith(".cif"):
                    bilayers_count += 1

    # 2) Trilayers Count (L=3 candidates generated)
    trilayers_dir = RESULTS_DIR / "3_layer"
    trilayers_count = 0
    if trilayers_dir.exists():
        for root, _, files in os.walk(trilayers_dir):
            for f in files:
                if f.lower().endswith(".cif"):
                    trilayers_count += 1

    # 3) Gather stable cifs recursively across any L-layer stable folders (negative_3_cifs, negative_4_cifs, etc.)
    stable_cifs = []
    
    # Scan project root for folders matching negative_*_cifs
    for item in os.listdir(PROJECT_DIR):
        item_path = PROJECT_DIR / item
        if item_path.is_dir() and item.startswith("negative_") and item != "negative_2_cifs":
            # Extract layer L from negative_L_cifs
            m = re.match(r"negative_(\d+)_cifs", item)
            if m:
                L = int(m.group(1))
                
                # Scan files in stable directory
                for root, _, files in os.walk(item_path):
                    for file in files:
                        if file.lower().endswith(".cif"):
                            file_path = os.path.join(root, file)
                            
                            # Parse metadata from filename and directory
                            # path like: negative_3_cifs/lower/r20/t1.2/t1.2_80_grid15.cif
                            norm_path = file_path.replace("\\", "/")
                            kind = "lower" if "/lower/" in norm_path else "upper"
                            
                            rot_match = re.search(r'/r(\d+)/', norm_path)
                            rot_val = int(rot_match.group(1)) if rot_match else 0
                            
                            disp_match = re.search(r'/t([0-9.]+)/', norm_path)
                            disp_val = float(disp_match.group(1)) if disp_match else 0.0
                            
                            # Parse rotation angle from filename
                            angle_match = re.search(r'_(\d+)_grid', file)
                            angle_val = int(angle_match.group(1)) if angle_match else 0
                            
                            stable_cifs.append({
                                "file_path": file_path,
                                "filename": file,
                                "L": L,
                                "kind": kind,
                                "rotation": rot_val,
                                "displacement": disp_val,
                                "angle": angle_val
                            })

    return {
        "bilayers_count": bilayers_count,
        "trilayers_count": trilayers_count,
        "stable_cifs_count": len(stable_cifs),
        "stable_cifs": stable_cifs
    }

# ================= STREAM RAW CIF ENDPOINT =================
@app.get("/api/view_cif")
def view_cif(path: str = Query(..., description="Absolute path to the CIF file")):
    """
    Streams raw CIF file text to be rendered by 3Dmol.js in the browser.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CIF file not found on disk.")
    
    # Security check: Ensure the file remains inside the project workspace directory
    try:
        real_path = Path(path).resolve()
        if not str(real_path).startswith(str(PROJECT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied. Path outside project workspace.")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path configuration.")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return Response(content=content, media_type="text/plain")

# ================= ASYNCHRONOUS PIPELINE STREAMS =================
class PipelineRequest(BaseModel):
    L: int
    mode: str  # 'generate' | 'extract' | 'predict' | 'all'
    cores: int
    angles: list[int]

@app.post("/api/run_pipeline")
async def run_pipeline_api(req: PipelineRequest):
    """
    Triggers generation/extraction/prediction steps for arbitrary layer count L.
    Streams logs dynamically to the frontend client console.
    """
    async def log_streamer():
        q = asyncio.Queue()

        def sync_logger(msg: str):
            # Safe wrapper to push messages to async event loop queue from worker threads
            loop.call_soon_threadsafe(q.put_nowait, msg + "\n")

        loop = asyncio.get_running_loop()

        # Target directories resolve
        parent_dir = PROJECT_DIR / f"negative_{req.L-1}_cifs"
        results_layer_dir = RESULTS_DIR / f"{req.L}_layer"
        csv_path = RESULTS_DIR / f"{req.L}_layer_features.csv"
        dest_dir = PROJECT_DIR / f"negative_{req.L}_cifs"

        # Worker method executed inside a separate background execution thread
        def worker():
            try:
                # 1) Resolve the highest available stable folder K_max
                k_max = 2
                while os.path.exists(PROJECT_DIR / f"negative_{k_max+1}_cifs"):
                    k_max += 1

                # 2) Run full intermediate pipelines if parent is missing
                for k in range(k_max + 1, req.L):
                    k_parent = PROJECT_DIR / f"negative_{k-1}_cifs"
                    k_results = RESULTS_DIR / f"{k}_layer"
                    k_csv = RESULTS_DIR / f"{k}_layer_features.csv"
                    k_dest = PROJECT_DIR / f"negative_{k}_cifs"

                    sync_logger(f"[SYSTEM] Parent stable folder negative_{k-1}_cifs is missing.")
                    sync_logger(f"[SYSTEM] Automatically running complete intermediate pipeline for L={k} first...")

                    # Generation for intermediate layer k
                    sync_logger(f"[SYSTEM] [Stage 1/3] Generating intermediate L={k} candidates...")
                    run_n_layer_generation(
                        L=k,
                        base_in=str(k_parent),
                        out_root=str(k_results),
                        chosen_angles=np.array(req.angles),
                        cores=req.cores,
                        logger=sync_logger
                    )

                    # Extraction for intermediate layer k
                    sync_logger(f"[SYSTEM] [Stage 2/3] Extracting intermediate L={k} features...")
                    run_n_layer_extraction(
                        L=k,
                        base_dir=str(k_results),
                        out_csv=str(k_csv),
                        cores=req.cores,
                        logger=sync_logger
                    )

                    # Prediction for intermediate layer k
                    sync_logger(f"[SYSTEM] [Stage 3/3] Classifying stable intermediate L={k} structures...")
                    run_n_layer_predictions(
                        L=k,
                        csv_path=str(k_csv),
                        model_path=str(MODEL_PATH),
                        metadata_path=str(VDW_CONFIG_PATH),
                        dest_dir=str(k_dest),
                        source_dir=str(k_results),
                        logger=sync_logger
                    )
                    sync_logger(f"[SYSTEM] Intermediate pipeline for L={k} complete. Created stable parent structures in {k_dest}.")

                # 3) Run requested pipeline mode for target layer count L
                parent_dir_target = PROJECT_DIR / f"negative_{req.L-1}_cifs"
                results_layer_dir_target = RESULTS_DIR / f"{req.L}_layer"
                csv_path_target = RESULTS_DIR / f"{req.L}_layer_features.csv"
                dest_dir_target = PROJECT_DIR / f"negative_{req.L}_cifs"

                if req.mode in ("generate", "all"):
                    sync_logger(f"[SYSTEM] Starting target L={req.L} structure generation...")
                    if not parent_dir_target.exists():
                        sync_logger(f"Error: Parent folder {parent_dir_target} not found.")
                        return
                    run_n_layer_generation(
                        L=req.L,
                        base_in=str(parent_dir_target),
                        out_root=str(results_layer_dir_target),
                        chosen_angles=np.array(req.angles),
                        cores=req.cores,
                        logger=sync_logger
                    )
                    sync_logger(f"[SYSTEM] Finished generating candidates for target L={req.L}")

                if req.mode in ("extract", "all"):
                    sync_logger(f"[SYSTEM] Starting target L={req.L} feature extraction...")
                    run_n_layer_extraction(
                        L=req.L,
                        base_dir=str(results_layer_dir_target),
                        out_csv=str(csv_path_target),
                        cores=req.cores,
                        logger=sync_logger
                    )

                if req.mode in ("predict", "all"):
                    sync_logger(f"[SYSTEM] Running target L={req.L} stability predictions...")
                    run_n_layer_predictions(
                        L=req.L,
                        csv_path=str(csv_path_target),
                        model_path=str(MODEL_PATH),
                        metadata_path=str(VDW_CONFIG_PATH),
                        dest_dir=str(dest_dir_target),
                        source_dir=str(results_layer_dir_target),
                        logger=sync_logger
                    )
                    sync_logger(f"[SYSTEM] Finished target L={req.L} predictions. Process complete!")

            except Exception as ex:
                sync_logger(f"[ERROR] Pipeline execution crashed: {ex}")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, "__SENTINEL_DONE__")

        # Start execution in a non-blocking thread pool
        import numpy as np
        asyncio.create_task(asyncio.to_thread(worker))

        # Yield stream logs chunk-by-chunk to HTTP Response Stream
        while True:
            log_line = await q.get()
            if log_line == "__SENTINEL_DONE__":
                break
            yield log_line

    return StreamingResponse(log_streamer(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
