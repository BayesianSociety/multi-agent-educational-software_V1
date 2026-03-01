import pathlib
import unittest


class RepoSmokeTests(unittest.TestCase):
    def test_required_paths_exist(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        required_files = [
            "REQUIREMENTS.md",
            "TEST.md",
            "AGENT_TASKS.md",
            "README.md",
            "RUNBOOK.md",
            "PROJECT_BRIEF.md",
            "PROJECT_BRIEF.yaml",
        ]
        required_dirs = ["design", "frontend", "backend", "tests"]
        for rel in required_files:
            self.assertTrue((root / rel).exists(), rel)
        for rel in required_dirs:
            self.assertTrue((root / rel).is_dir(), rel)


if __name__ == "__main__":
    unittest.main()
