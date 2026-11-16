import pytest

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.evaluation.syntax import patch_syntax


@pytest.mark.parametrize("replacement,valid", [("return 2", True), ("return (", False)])
def test_applied_python_patch_reports_syntax_independently(replacement, valid):
    snapshot = RepositorySnapshot(
        "fixture/repo", "a" * 40, (RepositoryFile("main.py", "def answer():\n    return 1\n"),)
    )
    patch = (
        "diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n"
        "@@ -1,2 +1,2 @@\n def answer():\n-    return 1\n+    " + replacement + "\n"
    )
    result = patch_syntax(snapshot, patch, applied=True)
    assert result == {"syntax_valid": valid, "syntax_checked_files": 1}
    skipped = patch_syntax(snapshot, patch, applied=False)
    assert skipped["syntax_valid"] is None and skipped["syntax_checked_files"] == 0
