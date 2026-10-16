# NanoStack 3D — Deployment & Hosting Architecture Plan
## Integration with Rutgers Amarel HPC and the Fang Lab Website

**Author / Maintainer:** Fang Lab, Rutgers University  
**Project:** NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor  
**Repository:** `Malav786/hong_agarose`  
**Date:** October 16, 2026  

---

## 1. Feasibility Verdict & Architectural Decision

### The Core Constraints
1. **Fang Lab Website (`https://sites.rutgers.edu/fang-lab/`):**
   * Hosted on **Sites@Rutgers**, a managed WordPress multisite environment operated by Rutgers Office of Information Technology (OIT).
   * **Verdict:** Cannot run Python, cannot run FastAPI/Uvicorn, cannot execute persistent background daemons, and cannot install arbitrary custom server-side plugins.
   * **Allowed capabilities:** Standard Gutenberg blocks (HTML, text, media, cards, buttons) and responsive links or verified `<iframe>` embeds.
2. **Rutgers Amarel HPC (`amarel3.hpc.rutgers.edu`, `p_hf267_1`):**
   * Institutional high-performance computing cluster managed by the Office of Advanced Research Computing (OARC).
   * Inbound network access is strictly firewalled and requires campus connection or Rutgers Cisco AnyConnect VPN with Duo 2FA.
   * **OARC Policy:** Prohibits running persistent, unauthenticated, public-facing internet servers directly on compute or login nodes. Outbound ephemeral reverse tunnels (e.g. cloudflared/ngrok) bypass institutional firewalls and violate OARC acceptable use.
   * **Optimal purpose:** High-throughput batch computation utilizing Prof. Hong's 8 dedicated 64-core nodes (512 cores, 256–512 GB RAM/node) and 24 TB GPFS storage (`/projects/f_hf267_1`).
3. **Public Application Showcase:**
   * NanoStack 3D requires a public endpoint with 3D molecular visualization (3Dmol.js) and lightweight ONNX model inference for external researchers.

### Recommended Architecture: Option A + C Hybrid (Decoupled Research & Demo Architecture)

```
[ Public Web Visitors & Researchers ]
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Fang Lab Website (sites.rutgers.edu/fang-lab/)       │
 │    Permalink: /forms/codes/ (NanoStack 3D Showcase)     │
 │    - Project Abstract & Methodology                     │
 │    - 3D Interactive Viewer (Embed or Direct Link)       │
 │    - Precomputed Stable Structures (L=2, 3, 4, 5)       │
 │    - Full Paper Citation & BibTeX Metadata              │
 └────────────────────────────┬────────────────────────────┘
                              │
               (Direct Link / Secure iframe)
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 2. Public Interactive Web Demo (Hugging Face / Web VM)  │
 │    - Hardened Docker container from repo Dockerfile     │
 │    - NANOSTACK_READ_ONLY=true (Public Read-Only Mode)   │
 │    - Automatic HTTPS, Zero Cloud Cost, No GPU Required  │
 │    - Serves 3Dmol.js viewer & ONNX candidate inference  │
 └─────────────────────────────────────────────────────────┘

 ══════════════ [ Rutgers Institutional Firewall & VPN ] ══════════════

 ┌─────────────────────────────────────────────────────────┐
 │ 3. Rutgers Amarel HPC Cluster (Dedicated Allocation)    │
 │    Partition: p_hf267_1 (8 Nodes, 64 Cores, 256-512 GB) │
 │    Storage:   /projects/f_hf267_1 (24 TB GPFS)          │
 │    OS:        RHEL 9 (amarel3 / amarel4 login nodes)    │
 │                                                         │
 │    Slurm Batch Workflow (`sbatch amarel_batch_...`):   │
 │    - High-throughput L=3, 4, 5 candidate generation     │
 │    - Numba JIT clash filtering across 64 CPU cores      │
 │    - Multi-interface feature extraction                 │
 │    - Output stored in /projects/f_hf267_1/runs/         │
 └─────────────────────────────────────────────────────────┘
```

---

## 2. Resource Specifications (Prof. Hong Amarel Allocation)

Verified from OARC provisioning telemetry:
* **Partition Name:** `p_hf267_1`
* **Dedicated Nodes (8 Total):**
  * **Bayonne Phase (4 Nodes):** `halk0030`, `halk0031`, `halk0032`, `halk0033` (64 CPUs, 256 GB RAM each)
  * **Clark Phase (4 Nodes):** `halk0110`, `halk0111`, `halk0112`, `halk0113` (64 CPUs, 512 GB RAM each)
  * **Total Compute Power:** 512 CPU cores, ~3 TB aggregate RAM.
* **Storage Allocation:**
  * Path: `/projects/f_hf267_1`
  * Quota: 24 TB (20 TB Bayonne + 4 TB Clark) on GPFS `projectsp` filesystem.
  * File count quota: 5,000,000 files.
* **Operating System:** Red Hat Enterprise Linux 9 (RHEL 9).
* **Login Hosts:** `amarel3.hpc.rutgers.edu`, `amarel4.hpc.rutgers.edu`.

---

## 3. Amarel HPC Setup & Execution Guide

### Step 1: Connect via Rutgers VPN & SSH to RHEL 9 Login Node
```bash
# Connect to Rutgers VPN (Cisco AnyConnect + Duo 2FA)
ssh <your_netid>@amarel3.hpc.rutgers.edu
```

### Step 2: Clone Repository to Project Storage
```bash
cd /projects/f_hf267_1
git clone https://github.com/Malav786/hong_agarose.git nanostack3d
cd nanostack3d
```

### Step 3: Load RHEL 9 Modules and Initialize Virtual Environment
```bash
module purge
module load python/3.11

# Create standard virtual environment in the project directory
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install package
pip install --upgrade pip
pip install -e ".[dev,web,ml-extra]"
```

### Step 4: Submit High-Throughput Batch Job
Use the dedicated batch Slurm script [amarel_batch_pipeline.slurm](file:///c:/Users/hong_admin/Desktop/published_code/amarel_batch_pipeline.slurm):

```bash
sbatch amarel_batch_pipeline.slurm
```

### Monitoring and Managing Slurm Jobs
```bash
# Check status of jobs on Prof. Hong's partition
squeue -p p_hf267_1

# Inspect real-time execution logs
tail -f /projects/f_hf267_1/logs/nanostack_batch_<job_id>.out

# Cancel a running job if necessary
scancel <job_id>
```

---

## 4. Public Web Demo Deployment (Zero Cost, Automatic HTTPS)

To make the interactive application publicly accessible without exposing Amarel or paying cloud infrastructure fees:

### Option A: Hugging Face Spaces (Recommended for Academic Research)
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Docker** SDK (Free CPU Tier: 2 vCPU, 16 GB RAM).
2. Push repository or configure repository mirror.
3. The existing [Dockerfile](file:///c:/Users/hong_admin/Desktop/published_code/Dockerfile) automatically binds to port 7860, runs under a non-root user (`useradd -m -u 1000 user`), and passes health checks (`/api/health`).
4. Set Space environment variable:
   ```
   NANOSTACK_READ_ONLY = true
   ```
5. Public URL: `https://huggingface.co/spaces/<organization>/nanostack3d` (Free, zero maintenance, permanent HTTPS).

---

## 5. Fang Lab Website Integration (`sites.rutgers.edu`)

**Target Page:** `https://sites.rutgers.edu/fang-lab/forms/codes/`

### WordPress Gutenberg Content Block Layout
In the WordPress dashboard for Sites@Rutgers, create or edit the page with the following sections:

#### Block 1: Headline & Scientific Abstract
```html
<div class="nanostack-header" style="padding: 20px 0; border-bottom: 2px solid #cc0033;">
  <h1 style="color: #cc0033; margin-bottom: 8px;">NanoStack 3D</h1>
  <p style="font-size: 1.2em; font-weight: 500; color: #333;">
    N-Layer Nanomaterial Architect & Stability Predictor
  </p>
  <p style="line-height: 1.6; color: #555;">
    An automated computational chemistry and machine learning framework for recursive generation,
    interfacial contact feature extraction, and thermodynamic stability prediction of multi-layer
    periodic molecular crystal configurations (agarose, polysaccharides, and layered 2D materials).
  </p>
</div>
```

#### Block 2: Interactive Application Embed / Launch Card
```html
<div style="background: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; padding: 24px; margin: 24px 0; text-align: center;">
  <h3 style="margin-top: 0;">Interactive 3D Molecular Viewer & Predictor</h3>
  <p>Explore curated stable bilayer and multilayer configurations rendered in real time using 3Dmol.js.</p>
  <div style="margin-top: 16px;">
    <a href="https://huggingface.co/spaces/fang-lab/nanostack3d" target="_blank" rel="noopener noreferrer" 
       style="background: #cc0033; color: #ffffff; padding: 12px 28px; text-decoration: none; border-radius: 4px; font-weight: bold; display: inline-block;">
      Launch Interactive NanoStack 3D Web App &rarr;
    </a>
  </div>
</div>
```

*(If Sites@Rutgers allows iframes for your user role, you can embed the responsive frame directly):*
```html
<iframe src="https://huggingface.co/spaces/fang-lab/nanostack3d" 
        width="100%" height="750px" frameborder="0" 
        style="border: 1px solid #ddd; border-radius: 8px;" 
        loading="lazy" allow="accelerometer; autoplay; clipboard-write">
</iframe>
```

#### Block 3: Scientific Performance & Methodology
* **Stacking Ensemble Model:** Random Forest + XGBoost + LightGBM + MLP with Sigmoid Platt Calibration.
* **Standard Random-Split Benchmark (80/20):** $MCC = 0.8421$, $ROC\text{-}AUC = 0.9789$, Accuracy $= 96.92\%$.
* **Rotation Group-Aware Cross-Validation (5-Fold):** $MCC = 0.7125 \pm 0.1174$ (evaluating generalization to unseen rotation angles).
* **Consensus Multi-Interface Rule:** An $L$-layer structure is classified as stable ($\Delta E \le 0$) if and only if **all $L-1$ interfaces** meet the calibrated thermodynamic stability threshold.

#### Block 4: Citation & BibTeX
```html
<div style="background: #f4f4f4; padding: 16px; border-left: 4px solid #cc0033; font-family: monospace; font-size: 0.9em;">
<pre>
@software{nanostack3d_2026,
  title  = {NanoStack 3D: N-Layer Nanomaterial Architect & Stability Predictor},
  author = {[AUTHOR'S FULL LEGAL OR PREFERRED PUBLICATION NAME]},
  year   = {2026},
  url    = {https://github.com/Malav786/hong_agarose}
}
</pre>
</div>
```

---

## 6. Security & Threat Mitigation Matrix

| Potential Risk | Threat Vector | Mitigation Implemented |
| :--- | :--- | :--- |
| **Path Traversal on Public Demo** | Manipulated query params like `/api/view_cif?path=../../win.ini` | `_validate_safe_cif_path` validates `Path.is_relative_to()`, allowlists directories (`data/`, `negative_*_cifs/`), and enforces `.cif` extension. |
| **Denial of Service (DoS) via Generation** | Unauthenticated triggering of expensive multi-threaded jobs | Server defaults to `NANOSTACK_READ_ONLY=true`. Pipeline route requires `X-Admin-Key` with fail-closed constant-time verification. |
| **HPC Network Intrusion** | Running public reverse tunnels on compute nodes | **Cloudflared tunnel removed from compute scripts**. HPC jobs run exclusively as batch processes; private interactive access restricted to VPN/SSH tunneling. |
| **Curated Data Corruption** | Concurrent writes or overwrite of benchmark CIF files | `atomic_write_cif` prevents truncated files; pipeline writes to isolated output directories without deleting reference folders. |

---

## 7. Immediate Next Steps for Deployment

1. **Confirm Professor / Lab Sign-Off on Hosting Option:**
   * Review the decoupled architecture (Fang Lab website link &rarr; Hugging Face Spaces read-only demo &rarr; Amarel Slurm batch backend).
2. **Deploy Public Read-Only Container:**
   * Create a free Hugging Face Space (or lab VM), push repository, and confirm `NANOSTACK_READ_ONLY=true` is set.
3. **Publish Fang Lab Page:**
   * Log into `https://sites.rutgers.edu/fang-lab/wp-admin/`, navigate to `forms/codes/`, and paste the provided showcase content and launch button.
