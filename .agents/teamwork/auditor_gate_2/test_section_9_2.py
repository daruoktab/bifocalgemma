import difflib
import hashlib
import sys
sys.path.insert(0, "src")
from bifocalgemma.data.offline_sft import editing_record

# 1. Test InstructCoder Adapter Simulation (Dual-key subject & instruction)
raw_instruct = {
    "subject": "Replace recursive factorial with an iterative loop.",
    "instruction": "Replace recursive factorial with an iterative loop.",
    "old_contents": "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n",
    "new_contents": "def fact(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n",
    "old_file": "solution.py",
    "new_file": "solution.py",
    "commit": "test_instruct_001",
    "repos": "likaixin/InstructCoder",
    "license": "open-research",
}
record_instruct = editing_record(raw_instruct, source="instructcoder")
assert record_instruct["target_patch"].startswith("diff --git a/solution.py b/solution.py\n")
assert "@@" in record_instruct["target_patch"]
assert record_instruct["group"] == "test_instruct_001"
print("[PASS] InstructCoder adapter successfully verified.")

# 2. Test GitHub Code Review Negative Pair Ingestion Rejection
raw_negative = {
    "subject": "Looks good to me.",
    "instruction": "Looks good to me.",
    "old_contents": "x = 10\n",
    "new_contents": "x = 10\n",  # before == after
    "old_file": "module.py",
    "new_file": "module.py",
    "commit": "pr_test_neg",
    "repos": "test/repo",
    "license": "mit",
}
try:
    editing_record(raw_negative, source="github_codereview")
    assert False, "Should have raised ValueError('empty_or_unchanged_code')"
except ValueError as e:
    assert str(e) == "empty_or_unchanged_code"
    print("[PASS] GitHub Code Review negative pair correctly rejected.")

# 3. Test POSIX Path Invariant Enforcement
raw_bad_path = {
    "subject": "Fix typo",
    "instruction": "Fix typo",
    "old_contents": "a = 1\n",
    "new_contents": "a = 2\n",
    "old_file": "src\\bad\\path.py",  # backslash violation
    "new_file": "src\\bad\\path.py",
    "commit": "c1",
    "repos": "r1",
    "license": "mit",
}
try:
    editing_record(raw_bad_path, source="test")
    assert False, "Should have raised ValueError('unsupported_path')"
except ValueError as e:
    assert str(e) == "unsupported_path"
    print("[PASS] Unsupported non-POSIX path correctly rejected.")
