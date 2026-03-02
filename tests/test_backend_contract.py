import pathlib
import unittest


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parents[1]

    def test_backend_scaffold_exists(self):
        required = [
            "backend/package.json",
            "backend/tsconfig.json",
            "backend/prisma/schema.prisma",
            "backend/src/server.ts",
            "backend/src/levels.ts",
        ]
        for rel in required:
            self.assertTrue((self.root / rel).exists(), rel)

    def test_schema_has_level_and_progress_models(self):
        schema = (self.root / "backend/prisma/schema.prisma").read_text(encoding="utf-8")
        self.assertIn("model Level", schema)
        self.assertIn("model Progress", schema)
        self.assertIn("blockedTiles", schema)
        self.assertIn("unlockedLevel", schema)

    def test_server_exposes_required_endpoints(self):
        server = (self.root / "backend/src/server.ts").read_text(encoding="utf-8")
        self.assertIn('app.get("/health"', server)
        self.assertIn('app.get("/api/levels"', server)
        self.assertIn('app.get("/api/levels/:id"', server)

    def test_mvp_level_seed_count(self):
        levels = (self.root / "backend/src/levels.ts").read_text(encoding="utf-8")
        # Deterministic sanity check: level ids 1..10 should exist in seed data.
        for level_id in range(1, 11):
            self.assertIn(f"id: {level_id}", levels)


if __name__ == "__main__":
    unittest.main()
