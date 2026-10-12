from __future__ import annotations
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

KERNEL = Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class ProtocolTests(unittest.TestCase):
    def test_manifest_schema_matches_runner_v2(self):
        template = json.loads((KERNEL / "experiments/MANIFEST_TEMPLATE.json").read_text())
        required = {
            "schema_version", "experiment_id", "run_id", "protocol_status", "gate_type",
            "git_commit", "git_branch", "git_available", "git_dirty", "git_status_porcelain",
            "execution_status", "gate_status", "return_code", "environment", "config",
        }
        self.assertEqual(template["schema_version"], 2)
        self.assertTrue(required.issubset(template))

    def test_e04_is_split_and_e2e_is_blocked(self):
        a = json.loads((KERNEL / "experiments/E04a_core_attention/config.json").read_text())
        b = json.loads((KERNEL / "experiments/E04b_end_to_end/config.json").read_text())
        self.assertEqual(a["automated_gate"]["timing_scope"], "core_attention_only")
        self.assertTrue(a["requires_clean_git"])
        self.assertTrue(b["status"].startswith("BLOCKED"))
        self.assertIn("FWHT", b["status"])

    def test_e07_is_observational(self):
        cfg = json.loads((KERNEL / "experiments/E07_long_context/config.json").read_text())
        self.assertEqual(cfg["gate_type"], "observational")
        self.assertEqual(cfg["depends_on"], ["E04a_core_attention"])

    def test_core_launcher_has_no_item_sync(self):
        from ofd_kernel.runtime.launcher import triton_paged_decode_core
        source = inspect.getsource(triton_paged_decode_core)
        self.assertNotIn(".item(", source)

    def test_runner_uses_core_timing_for_e04a_and_e07(self):
        runner = load_module(KERNEL / "tools/run_experiment.py", "ofd_runner_test")
        c1 = runner.commands_for("E04a_core_attention", Path("/tmp/x"))[0]
        c2 = runner.commands_for("E07_long_context", Path("/tmp/x"))[0]
        self.assertIn("core-variants", c1)
        self.assertIn("long-core", c2)
        self.assertEqual(runner.gate_from_success({"gate_type": "observational"}), "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
