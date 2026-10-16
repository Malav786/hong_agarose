# Security Policy

## Supported Versions

Security fixes and defensive updates are applied to the active release line:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

If you discover an application security issue, path traversal vulnerability, or potential data exposure in NanoStack 3D:

1. **Do not open a public GitHub issue.**
2. Send a confidential report via GitHub Private Vulnerability Reporting on the repository's 'Security' tab.
3. Include:
   * A description of the vulnerability.
   * Reproduction steps or a proof-of-concept request.
   * Potential impact on research data or server confidentiality.

We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline.

---

## Security Architecture & Deployment Defenses

NanoStack 3D implements the following defensive controls:

1. **Path Traversal Protection**: Endpoints `/api/view_cif` and `/api/download_cif` strictly enforce:
   * Path containment inside approved subdirectories (`data/`, `results/`, `negative_*_cifs/`).
   * Explicit `.cif` file extension whitelisting.
   * Complete rejection of traversal tokens (`../`), null bytes, symlink escapes, and non-cif system files.
2. **Public Demo Read-Only Guard**: The server defaults to read-only demonstration mode (`NANOSTACK_READ_ONLY=true`).
   * Destructive or resource-intensive generation operations via `/api/run_pipeline` are blocked unless authenticated with `X-Admin-Key`.
   * Curated research directories (`negative_*_cifs/`) are protected against deletion.
3. **Concurrency Controls**: Pipeline runs acquire an asynchronous execution lock to prevent parallel worker exhaustion and denial-of-service (DoS).
