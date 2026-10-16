#!/bin/bash
# ==============================================================================
# NanoStack 3D: High-Throughput N-Layer Batch Execution on Rutgers Amarel HPC
# Partition: p_hf267_1 (Prof. Hong Dedicated Allocation - 8 Nodes, 64 Cores/Node)
# Storage:   /projects/f_hf267_1 (24 TB GPFS Filesystem)
# ==============================================================================

#SBATCH --job-name=nanostack_batch
#SBATCH --partition=p_hf267_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/f_hf267_1/logs/%x_%j.out
#SBATCH --error=/projects/f_hf267_1/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@institution.edu  # Default placeholder; overridden automatically by submit_batch.sh

echo "=========================================================="
echo "NanoStack 3D High-Throughput Batch Job Started"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Running on:    $(hostname)"
echo "Partition:     ${SLURM_JOB_PARTITION} (Prof. Hong Allocation)"
echo "Cores Alloc:   ${SLURM_CPUS_PER_TASK}"
echo "Start Time:    $(date)"
echo "=========================================================="

# 1. Base Paths & Project Storage
PROJECT_STORAGE="/projects/f_hf267_1"
PROJECT_DIR="${PROJECT_STORAGE}/code/hong_agarose"
LOG_DIR="${PROJECT_STORAGE}/logs"
mkdir -p "${LOG_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
    PROJECT_DIR="$(pwd)"
fi

cd "${PROJECT_DIR}" || exit 1

# Load private environment variables if present (.env is untracked by git)
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

# 2. Virtual Environment Activation
VENV_PATH="${PROJECT_STORAGE}/envs/nanostack_venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
elif [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
else
    echo "[ERROR] Virtual environment not found at ${VENV_PATH}"
    exit 1
fi

echo "Active Python: $(which python3)"
python3 --version

# 4. CPU Threading Controls (Prevent Thread Contention / Oversubscription)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=64
export PYTHONUNBUFFERED=1

# 5. Output Workspace Setup (Job-Isolated to protect curated benchmarks)
JOB_OUT="${PROJECT_STORAGE}/runs/job_${SLURM_JOB_ID}_L3"
mkdir -p "${JOB_OUT}"

echo "[START] Running NanoStack 3D Generation & Consensus Stability Pipeline..."
echo "Output Directory: ${JOB_OUT}"

# 6. Execute Batch Pipeline via CLI
# Adjust target layer (--layer) and input parent as required (L=3, L=4, L=5)
python -m script.core.cli run-pipeline \
    --layer 3 \
    --mode all \
    --input "${PROJECT_DIR}/negative_2_cifs" \
    --output-dir "${JOB_OUT}" \
    --cores 64

EXIT_CODE=$?

echo "=========================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[SUCCESS] Batch job finished cleanly."
    echo "Results directory: ${JOB_OUT}"
    echo "Stable structures: ${JOB_OUT}/stable_cifs"
else
    echo "[FAILURE] Batch job exited with code ${EXIT_CODE}"
fi
echo "End Time: $(date)"
echo "=========================================================="

exit ${EXIT_CODE}
