import pathlib
import re
import unittest


class DocsContractTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parents[1]

    def test_test_md_required_headings(self):
        test_md = (self.root / "TEST.md").read_text(encoding="utf-8")
        self.assertIn("# How to run tests", test_md)
        self.assertIn("# Environments", test_md)

    def test_test_md_has_executable_code_block(self):
        test_md = (self.root / "TEST.md").read_text(encoding="utf-8")
        code_blocks = re.findall(r"```(?:bash)?\n(.*?)```", test_md, flags=re.DOTALL)
        self.assertTrue(code_blocks, "TEST.md must contain at least one fenced code block")
        self.assertTrue(
            any("python3 -m unittest discover -s tests -p 'test_*.py'" in block for block in code_blocks),
            "TEST.md must include executable deterministic unittest command",
        )


if __name__ == "__main__":
    unittest.main()
