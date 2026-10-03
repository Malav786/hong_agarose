#!/bin/bash
#SBATCH --job-name=nanostack3d
#SBATCH --partition=p_hf267_1
#SBATCH --nodes=1
#SBATCH --nodelist=halk0113
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=/projects/f_hf267_1/nanostack_service_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mjc701@scarletmail.rutgers.edu,mchamp.2509@gmail.com


echo "=========================================================="
echo "Starting NanoStack 3D Service on Amarel Node: $(hostname)"
echo "Partition: p_hf267_1 (Prof. Hong Dedicated Allocation)"
echo "Storage:   /projects/f_hf267_1"
echo "Date: $(date)"
echo "=========================================================="

PROJECT_DIR="/projects/f_hf267_1/hong_agarose"
LOG_FILE="/projects/f_hf267_1/nanostack_service_${SLURM_JOB_ID}.log"

# 1. Initialize Lmod / Environment Modules on RHEL 9
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

# 2. Load Python (RHEL9 Amarel modules)
module purge 2>/dev/null
module load python/3.11 2>/dev/null || \
module load python/3.10 2>/dev/null || \
module load python/3.12 2>/dev/null || \
module load miniconda3  2>/dev/null || \
module load anaconda    2>/dev/null

echo "Active Python: $(which python3)"
python3 --version

# 3. Move into project directory on shared storage
cd "$PROJECT_DIR" || { echo "[ERROR] Project directory not found: $PROJECT_DIR"; exit 1; }

# 4. Clean up broken virtualenv (e.g. wrong Python version)
if [ -d ".venv" ]; then
    VENV_PY_VER=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ "$VENV_PY_VER" = "3.9" ] || [ ! -f ".venv/bin/uvicorn" ]; then
        echo "[CLEANUP] Removing incompatible virtualenv (Python $VENV_PY_VER)..."
        rm -rf .venv
    fi
fi

# 5. Create or activate virtual environment
if [ ! -d ".venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "[OK] Python env ready: $(which python)"

# 6. Download cloudflared if not already present
CLOUDFLARED="$PROJECT_DIR/cloudflared"
if [ ! -f "$CLOUDFLARED" ]; then
    echo "[SETUP] Downloading cloudflared binary..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
         -O "$CLOUDFLARED"
    chmod +x "$CLOUDFLARED"
fi

# 7. Launch FastAPI server
echo "[START] Launching NanoStack 3D Uvicorn server on port 8000..."
python -m uvicorn final_system.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
echo "[INFO]  Server PID: $SERVER_PID"

# Wait for server to bind
sleep 8

# Verify server is actually running before starting tunnel
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] Uvicorn failed to start. Check logs above."
    exit 1
fi
echo "[OK] Server is running."

# 8. Start Cloudflare Quick Tunnel
echo "=========================================================="
echo "  Starting Cloudflare Tunnel..."
echo "  The public URL will appear on the next line:"
echo "=========================================================="

"$CLOUDFLARED" tunnel --url http://127.0.0.1:8000 2>&1 | tee -a "$LOG_FILE"

# 9. Cleanup on exit
echo "[STOP] Tunnel ended. Shutting down server (PID $SERVER_PID)..."
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo "[DONE] $(date)"
