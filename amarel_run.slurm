#!/bin/bash
#SBATCH --job-name=nanostack3d
#SBATCH --partition=p_hf267_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=nanostack_service_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@institution.edu  # Set your email address or override at submit time

echo "=========================================================="
echo "Starting NanoStack 3D Service on Amarel Node: $(hostname)"
echo "Partition: p_hf267_1 (Prof. Hong Dedicated Allocation)"
echo "Date: $(date)"
echo "=========================================================="

# Auto-detect project directory from where the job was submitted
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR" || { echo "[ERROR] Failed to cd to $PROJECT_DIR"; exit 1; }
LOG_FILE="${PROJECT_DIR}/nanostack_service_${SLURM_JOB_ID}.log"

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

# 3. Ensure we are in project directory
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

# 6. Secure Access Note:
# For private lab access from campus or via Rutgers VPN, use policy-compliant
# SSH local port forwarding from your workstation:
#   ssh -N -L 8000:$(hostname):8000 netid@amarel3.hpc.rutgers.edu
# Then open http://localhost:8000 in your browser.
#
# (Note: Ephemeral public tunnels such as cloudflared on institutional HPC
# compute nodes may trigger network security alerts and are restricted by OARC).

# 7. Launch FastAPI server
echo "[START] Launching NanoStack 3D Uvicorn server on port 8000..."
echo "[INFO]  Compute Node: $(hostname)"
echo "[INFO]  Secure SSH Tunnel command for lab members:"
echo "        ssh -N -L 8000:$(hostname):8000 <netid>@amarel3.hpc.rutgers.edu"
python -m uvicorn final_system.main:app --host 127.0.0.1 --port 8000

echo "[DONE] Server shut down at $(date)"
