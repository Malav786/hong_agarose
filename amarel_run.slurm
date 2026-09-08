#!/bin/bash
#SBATCH --job-name=nanostack3d
#SBATCH --partition=p_hf267_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=30-00:00:00
#SBATCH --output=nanostack_service_%j.log
#SBATCH --error=nanostack_service_%j.err

echo "=========================================================="
echo "Starting NanoStack 3D Service on Amarel Node: $(hostname)"
echo "Partition: p_hf267_1 (Owner Allocation)"
echo "Date: $(date)"
echo "=========================================================="

# 1. Load Python module on RHEL 9
if command -v module &> /dev/null; then
    module purge
    module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load miniconda.3 2>/dev/null
fi

# 2. Initialize and activate virtual environment
if [ ! -d ".venv" ]; then
    echo "[SETUP] Creating python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 3. Ensure Cloudflare Tunnel binary is available locally
if [ ! -f "./cloudflared" ]; then
    echo "[SETUP] Downloading cloudflared standalone binary..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

# 4. Start the FastAPI dashboard server on local port 8000
echo "[START] Launching NanoStack 3D Uvicorn Server..."
python -m uvicorn final_system.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Allow server 5 seconds to bind
sleep 5

# 5. Start Cloudflare Tunnel to create a permanent global HTTPS link
echo "=========================================================="
echo "Starting Global Public Tunnel via Cloudflare..."
echo "Your live public link will appear below:"
echo "=========================================================="

./cloudflared tunnel --url http://127.0.0.1:8000

# Cleanup when job terminates
kill $SERVER_PID
