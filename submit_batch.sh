#!/bin/bash
# ==============================================================================
# Helper to submit NanoStack 3D batch job to Slurm reading private email from .env
# Usage: ./submit_batch.sh [extra sbatch arguments...]
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ -f "${ENV_FILE}" ]; then
    EMAIL=$(grep -E '^[[:space:]]*SLURM_NOTIFICATION_EMAIL=' "${ENV_FILE}" | cut -d= -f2 | tr -d ' "' | tr -d '\r')
fi

if [ -n "${EMAIL}" ] && [ "${EMAIL}" != "user@institution.edu" ]; then
    echo "Submitting job with notification email: ${EMAIL}"
    sbatch --mail-user="${EMAIL}" "${SCRIPT_DIR}/amarel_batch_pipeline.slurm" "$@"
else
    echo "Submitting with default script settings (no custom .env email detected)"
    sbatch "${SCRIPT_DIR}/amarel_batch_pipeline.slurm" "$@"
fi
