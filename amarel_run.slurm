#!/bin/bash
#SBATCH --job-name=nanostack3d
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=nanostack_service_%j.log

echo "=========================================================="
echo "Starting NanoStack 3D Service on Amarel Node: $(hostname)"
echo "Partition: main (General Amarel Allocation)"
echo "Date: $(date)"
echo "=========================================================="

# 1. Initialize Lmod / Environment Modules on RHEL 9
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

# 2. Load modern Python module (Python 3.10, 3.11, or 3.12)
module purge 2>/dev/null
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python/3.12 2>/dev/null || module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null

echo "Active Python: $(which python3)"
python3 --version

# 3. Clean up any broken/outdated virtual environment if Python 3.9 was used
if [ -d ".venv" ]; then
    VENV_PY_VER=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ "$VENV_PY_VER" = "3.9" ] || [ ! -f ".venv/bin/uvicorn" ]; then
        echo "[CLEANUP] Removing incompatible or incomplete virtualenv ($VENV_PY_VER)..."
        rm -rf .venv
    fi
fi

# 4. Initialize and activate modern virtual environment
if [ ! -d ".venv" ]; then
    echo "[SETUP] Creating modern python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 5. Ensure Cloudflare Tunnel binary is available locally
if [ ! -f "./cloudflared" ]; then
    echo "[SETUP] Downloading cloudflared standalone binary..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

# 6. Start the FastAPI dashboard server on local port 8000
echo "[START] Launching NanoStack 3D Uvicorn Server..."
python -m uvicorn final_system.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Allow server 5 seconds to bind
sleep 5

# 7. Start Cloudflare Tunnel to create a permanent global HTTPS link
echo "=========================================================="
echo "Starting Global Public Tunnel via Cloudflare..."
echo "Your live public link will appear below:"
echo "=========================================================="

./cloudflared tunnel --url http://127.0.0.1:8000 2>&1

# Cleanup when job terminates
kill $SERVER_PID
