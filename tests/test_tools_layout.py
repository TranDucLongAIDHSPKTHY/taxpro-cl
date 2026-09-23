import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ToolsLayoutTests(unittest.TestCase):
    def test_command_modules_resolve_the_repository_root(self):
        modules = (
            "tools.data.download",
            "tools.data.build_splits",
            "tools.data.prepare_metadata",
            "tools.data.metadata_manifest",
            "tools.protocol.build",
            "tools.protocol.validate_determinism",
            "tools.protocol.validate_gate",
            "tools.analysis.compile_runs",
            "tools.analysis.compile_ablation_sweep",
            "tools.repository.preflight",
        )
        for name in modules:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertEqual(module.ROOT, ROOT)

    def test_legacy_cli_paths_are_removed(self):
        obsolete = (
            "download_data.py",
            "prepare_metadata.py",
            "week3_protocol.py",
            "preflight.py",
            "compile_results.py",
            "running",
        )
        for relative in obsolete:
            with self.subTest(path=relative):
                self.assertFalse((ROOT / "tools" / relative).exists())


if __name__ == "__main__":
    unittest.main()
