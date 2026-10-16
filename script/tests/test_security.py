"""
Automated Security Test Suite for NanoStack 3D.
Validates:
1. Path traversal rejection on /api/view_cif and /api/download_cif.
2. Protection against arbitrary file disclosure (.git, configs, Python sources).
3. Rejection of null bytes, non-CIF extensions, and malformed paths.
4. Public demonstration read-only enforcement on POST /api/run_pipeline.
5. Administrative authentication requirements and invalid token rejection.
6. Target dataset immutability (curated negative_*_cifs folders are preserved).
7. Health check endpoint status and telemetry.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project and final_system are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_SYSTEM_DIR = PROJECT_ROOT / "final_system"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FINAL_SYSTEM_DIR))

# Ensure read-only mode is active during testing by default
os.environ["NANOSTACK_READ_ONLY"] = "true"
os.environ["NANOSTACK_ADMIN_KEY"] = "test_secret_admin_key_12345"

from final_system.main import app, _validate_safe_cif_path


class TestSecurityEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_cif = PROJECT_ROOT / "data" / "r0" / "t0" / "t0_0.cif"

    def test_01_health_endpoint(self):
        """Verify health check returns healthy status and read-only mode telemetry."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["read_only"])
        self.assertIn("model_backend", data)
        self.assertIn("timestamp", data)

    def test_02_valid_cif_access(self):
        """Verify that legitimate CIF files in allowed directories are accessible."""
        self.assertTrue(self.valid_cif.exists(), f"Valid CIF missing at: {self.valid_cif}")
        
        # Test view endpoint
        resp_view = self.client.get(f"/api/view_cif?path={self.valid_cif}")
        self.assertEqual(resp_view.status_code, 200)
        self.assertIn("data_", resp_view.text)

        # Test download endpoint
        resp_dl = self.client.get(f"/api/download_cif?path={self.valid_cif}")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertEqual(resp_dl.headers["content-type"], "chemical/x-cif")

    def test_03_path_traversal_rejection(self):
        """Verify that directory traversal attempts outside the project root are blocked."""
        traversal_attempts = [
            "../../../../../../windows/win.ini",
            "../../../../../../etc/passwd",
            "..\\..\\..\\..\\windows\\system32\\cmd.exe",
            str(PROJECT_ROOT.parent / "secret.txt"),
            "../outside.cif",
        ]
        for bad_path in traversal_attempts:
            resp = self.client.get(f"/api/view_cif?path={bad_path}")
            self.assertIn(
                resp.status_code, [400, 403, 404],
                f"Path traversal not blocked for {bad_path}: status {resp.status_code}"
            )

    def test_04_internal_sensitive_file_access_blocked(self):
        """Verify that non-CIF internal repository files cannot be viewed or downloaded."""
        forbidden_files = [
            PROJECT_ROOT / "final_system" / "main.py",
            PROJECT_ROOT / "amarel_run.slurm",
            PROJECT_ROOT / "requirements.txt",
            PROJECT_ROOT / "results" / "model_threshold.json",
            PROJECT_ROOT / ".gitignore",
        ]
        for fpath in forbidden_files:
            resp_view = self.client.get(f"/api/view_cif?path={fpath}")
            self.assertEqual(
                resp_view.status_code, 403,
                f"Sensitive file access was not blocked with 403: {fpath}"
            )

            resp_dl = self.client.get(f"/api/download_cif?path={fpath}")
            self.assertEqual(
                resp_dl.status_code, 403,
                f"Sensitive file download was not blocked with 403: {fpath}"
            )

    def test_05_null_bytes_and_malformed_inputs(self):
        """Verify that null bytes and malformed paths are rejected with bad request."""
        # 1. Direct validator function raises HTTPException 400 on null byte
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            _validate_safe_cif_path(f"{self.valid_cif}\x00.cif")
        self.assertEqual(ctx.exception.status_code, 400)

        # 2. Malformed / URL-encoded inputs via client
        malformed = [
            f"{self.valid_cif}%00.cif",
            "",
            "   ",
        ]
        for bad_input in malformed:
            resp = self.client.get(f"/api/view_cif?path={bad_input}")
            self.assertIn(resp.status_code, [400, 403, 404, 422])

    def test_06_run_pipeline_read_only_protection(self):
        """Verify that unauthenticated pipeline runs in read-only demo mode are rejected."""
        payload = {
            "L": 3,
            "mode": "predict",
            "cores": 2,
            "angles": [0, 20]
        }
        # Request with no admin key
        resp = self.client.post("/api/run_pipeline", json=payload)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("disabled in public demonstration mode", resp.json()["detail"])

        # Request with invalid admin key
        resp_bad_key = self.client.post(
            "/api/run_pipeline",
            json=payload,
            headers={"X-Admin-Key": "wrong_password_attempt"}
        )
        self.assertEqual(resp_bad_key.status_code, 403)

    def test_07_run_pipeline_input_validation(self):
        """Verify that invalid parameters (layer out of bounds, invalid mode) are rejected."""
        # Layer too low (<3)
        resp1 = self.client.post(
            "/api/run_pipeline",
            json={"L": 1, "mode": "all", "cores": 2, "angles": [0]},
            headers={"X-Admin-Key": "test_secret_admin_key_12345"}
        )
        self.assertEqual(resp1.status_code, 422)

        # Layer too high (>10)
        resp2 = self.client.post(
            "/api/run_pipeline",
            json={"L": 99, "mode": "all", "cores": 2, "angles": [0]},
            headers={"X-Admin-Key": "test_secret_admin_key_12345"}
        )
        self.assertEqual(resp2.status_code, 422)

        # Invalid mode
        resp3 = self.client.post(
            "/api/run_pipeline",
            json={"L": 3, "mode": "destroy_all", "cores": 2, "angles": [0]},
            headers={"X-Admin-Key": "test_secret_admin_key_12345"}
        )
        self.assertEqual(resp3.status_code, 400)

    def test_08_curated_data_immutability(self):
        """Verify that curated negative_*_cifs folders are not mutated or deleted."""
        curated_l2 = PROJECT_ROOT / "negative_2_cifs"
        self.assertTrue(curated_l2.exists(), "Curated negative_2_cifs folder must exist!")
        cifs_before = list(curated_l2.glob("**/*.cif"))
        self.assertEqual(len(cifs_before), 300, "Expected exactly 300 curated bilayer CIF files.")

        # Attempt forbidden run
        self.client.post("/api/run_pipeline", json={"L": 3, "mode": "predict", "cores": 2, "angles": [0]})

        # Verify folder and counts remain 100% intact
        cifs_after = list(curated_l2.glob("**/*.cif"))
        self.assertEqual(len(cifs_after), 300, "Curated CIF count changed unexpectedly!")

    def test_09_encoded_path_traversal_rejection(self):
        """Verify that URL-encoded traversal sequences are blocked."""
        encoded_attempts = [
            "%2e%2e%2f%2e%2e%2fmain.py",
            "data%2f..%2f..%2frequirements.txt",
            "%2e%2e%5c%2e%2e%5camarel_run.slurm",
        ]
        for bad_enc in encoded_attempts:
            resp = self.client.get(f"/api/view_cif?path={bad_enc}")
            self.assertIn(resp.status_code, [400, 403, 404])

    def test_10_admin_fail_closed_configuration(self):
        """Verify that if read-only mode is false but admin key is unset, system fails closed."""
        import final_system.main as main_mod
        orig_read_only = main_mod.READ_ONLY_MODE
        orig_admin_key = main_mod.ADMIN_API_KEY
        try:
            main_mod.READ_ONLY_MODE = False
            main_mod.ADMIN_API_KEY = ""  # Unconfigured key

            payload = {"L": 3, "mode": "predict", "cores": 2, "angles": [0]}
            resp = self.client.post("/api/run_pipeline", json=payload, headers={"X-Admin-Key": "any_key"})
            self.assertEqual(resp.status_code, 500)
            self.assertIn("fails closed", resp.json()["detail"])
        finally:
            main_mod.READ_ONLY_MODE = orig_read_only
            main_mod.ADMIN_API_KEY = orig_admin_key


if __name__ == "__main__":
    unittest.main()
