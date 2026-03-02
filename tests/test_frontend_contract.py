import pathlib
import unittest


class FrontendContractTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parents[1]

    def test_frontend_files_exist(self):
        required = [
            "frontend/package.json",
            "frontend/src/app/page.tsx",
            "frontend/src/app/layout.tsx",
            "frontend/src/app/globals.css",
            "frontend/src/lib/gameEngine.ts",
            "frontend/src/lib/types.ts",
            "frontend/src/api.ts",
        ]
        for rel in required:
            self.assertTrue((self.root / rel).exists(), rel)

    def test_required_endpoint_usage(self):
        api = (self.root / "frontend/src/api.ts").read_text(encoding="utf-8")
        self.assertIn("/api/levels", api)
        self.assertIn("/api/levels/${id}", api)

    def test_execution_model_contract(self):
        engine = (self.root / "frontend/src/lib/gameEngine.ts").read_text(encoding="utf-8")
        self.assertIn("compileBlocks", engine)
        self.assertIn("runInstruction", engine)
        self.assertIn("evaluateProgram", engine)

    def test_accessibility_cues_present(self):
        page = (self.root / "frontend/src/app/page.tsx").read_text(encoding="utf-8")
        css = (self.root / "frontend/src/app/globals.css").read_text(encoding="utf-8")
        self.assertIn("aria-live", page)
        self.assertIn("aria-label", page)
        self.assertIn(":focus-visible", css)


if __name__ == "__main__":
    unittest.main()
