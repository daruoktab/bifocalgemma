import ast
import hashlib
import re
import sys
import textwrap
from pathlib import PurePosixPath

sys.path.insert(0, "src")
from bifocalgemma.data.offline_sft import editing_record

# --- Section 3.2: InstructCoder Adapter ---
def map_instructcoder_to_bifocal(raw: dict, index: int) -> dict | None:
    instruction = (raw.get("instruction") or "").strip()
    before_code = raw.get("input") or ""
    after_code = raw.get("output") or ""
    
    # Quality Filter: reject scraping artifacts and malformed data
    if "Copy code" in before_code or "Copy code" in after_code:
        return None
    if not instruction or not before_code or not after_code or before_code == after_code:
        return None
    
    # Syntax validation on standalone Python functions
    try:
        ast.parse(before_code)
        ast.parse(after_code)
    except SyntaxError:
        return None
    
    path = "solution.py"
    pair_hash = hashlib.sha256((before_code + "\0" + after_code).encode("utf-8")).hexdigest()
    commit = f"instructcoder_{pair_hash[:16]}"
    
    raw_record = {
        "subject": instruction,      # Satisfies offline_sft.py:13 when source != "editpackft"
        "instruction": instruction,  # Retained for cross-compatibility
        "old_contents": before_code,
        "new_contents": after_code,
        "old_file": path,
        "new_file": path,
        "commit": commit,
        "repos": "likaixin/InstructCoder",
        "license": "open-research",
    }
    return editing_record(raw_record, source="instructcoder")

# --- Section 3.3: GitHub Code Review Adapter ---
def sanitize_posix_path(raw_path: str | None) -> str:
    if not raw_path or not isinstance(raw_path, str):
        return "module.py"
    clean = raw_path.replace("\\", "/").lstrip("/")
    parts = [p for p in clean.split("/") if p and not p.startswith(".")]
    clean_path = "/".join(parts)
    clean_path = re.sub(r"[\s:]", "_", clean_path)
    return clean_path or "module.py"

def map_codereview_to_bifocal(raw: dict, cceval_repo_blacklist: set[str]) -> dict | None:
    # 1. Reject non-Python, negative pairs, and CCEval overlapping repositories
    if raw.get("language") != "Python" or raw.get("is_negative", False):
        return None
    repo_name = (raw.get("repo_name") or "").lower().strip()
    if repo_name in cceval_repo_blacklist or any(repo_name.endswith(f"/{r}") for r in cceval_repo_blacklist):
        return None
    
    before_code = raw.get("before_code") or ""
    after_code = raw.get("after_code") or ""
    instruction = (raw.get("reviewer_comment") or "").strip()
    
    if not before_code or not after_code or before_code == after_code or not instruction:
        return None
        
    # Syntax check on code chunks: dedent first to prevent IndentationError on nested snippets
    try:
        ast.parse(textwrap.dedent(before_code))
        ast.parse(textwrap.dedent(after_code))
    except (SyntaxError, IndentationError):
        return None
        
    path = sanitize_posix_path(raw.get("file_path"))
    pr_num = raw.get("pr_number", 0)
    commit = f"{repo_name.replace('/', '_')}_pr_{pr_num}"
    
    raw_record = {
        "subject": instruction,      # Satisfies offline_sft.py:13 when source != "editpackft"
        "instruction": instruction,  # Retained for cross-compatibility
        "old_contents": before_code,
        "new_contents": after_code,
        "old_file": path,
        "new_file": path,
        "commit": commit,
        "repos": raw.get("repo_name", "unknown"),
        "license": "permissive-oss",
    }
    return editing_record(raw_record, source="github_codereview")

# --- Section 3.4: NextCoder Adapter ---
def map_nextcoder_to_bifocal(raw: dict, index: int) -> dict | None:
    prompt = raw.get("prompt", "")
    completion = raw.get("completion", "")
    
    # Extract code fences
    m_prompt = re.search(r"```python\s*(.*?)\s*```", prompt, re.DOTALL)
    if not m_prompt:
        return None
    before_code = m_prompt.group(1).strip()
    
    # Extract instruction text preceding code fence
    prompt_text = prompt[:m_prompt.start()].strip()
    prefix = "Rewrite the given python program as per the following instruction.\n"
    if prompt_text.startswith(prefix):
        prompt_text = prompt_text[len(prefix):]
    suffix = "\nWrite the entire code and no other text in the response."
    if prompt_text.endswith(suffix):
        prompt_text = prompt_text[:-len(suffix)]
    instruction = prompt_text.strip()
    
    m_comp = re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
    after_code = m_comp.group(1).strip() if m_comp else completion.strip()
    
    if not before_code or not after_code or before_code == after_code or not instruction:
        return None
        
    # Syntax validation on standalone program pairs
    try:
        ast.parse(before_code)
        ast.parse(after_code)
    except SyntaxError:
        return None
        
    path = "solution.py"
    # Derive commit from program seed hash to prevent train/dev leakage across prompt variants
    seed_id = raw.get("seed_id") or hashlib.sha256(before_code.encode("utf-8")).hexdigest()[:16]
    commit = f"nextcoder_seed_{seed_id}"
    
    raw_record = {
        "subject": instruction,      # Satisfies offline_sft.py:13 when source != "editpackft"
        "instruction": instruction,  # Retained for cross-compatibility
        "old_contents": before_code,
        "new_contents": after_code,
        "old_file": path,
        "new_file": path,
        "commit": commit,
        "repos": "microsoft/NextCoderDataset",
        "license": "mit",
    }
    return editing_record(raw_record, source="nextcoder")


# --- RUN EMPIRICAL ASSERTIONS ---

# Test InstructCoder
ic_sample = {
    "instruction": "Add type annotations to function.",
    "input": "def add(a, b):\n    return a + b\n",
    "output": "def add(a: int, b: int) -> int:\n    return a + b\n",
}
res_ic = map_instructcoder_to_bifocal(ic_sample, 0)
assert res_ic is not None
assert res_ic["instruction"] == "Add type annotations to function."
assert res_ic["target_patch"].startswith("diff --git a/solution.py b/solution.py")
print("[PASS] map_instructcoder_to_bifocal")

# Test InstructCoder scraping filter
ic_bad = {
    "instruction": "Fix",
    "input": "Copy code\ndef foo(): pass",
    "output": "def foo(): pass",
}
assert map_instructcoder_to_bifocal(ic_bad, 1) is None
print("[PASS] map_instructcoder_to_bifocal rejects Copy code")

# Test GitHub Code Review with leading indent (needs textwrap.dedent)
cr_indented = {
    "language": "Python",
    "is_negative": False,
    "repo_name": "clean/repo",
    "file_path": "src/utils.py",
    "pr_number": 123,
    "reviewer_comment": "Check if item is None before accessing item.val",
    "before_code": "        if item:\n            return item.val\n        return 0\n",
    "after_code": "        if item is not None:\n            return item.val\n        return 0\n",
}
res_cr = map_codereview_to_bifocal(cr_indented, {"bad/repo"})
assert res_cr is not None
assert res_cr["target_patch"].startswith("diff --git a/src/utils.py b/src/utils.py")
print("[PASS] map_codereview_to_bifocal handled indented chunk via textwrap.dedent")

# Test GitHub Code Review blacklist rejection
cr_blacklisted = dict(cr_indented)
cr_blacklisted["repo_name"] = "bad/repo"
assert map_codereview_to_bifocal(cr_blacklisted, {"bad/repo"}) is None
print("[PASS] map_codereview_to_bifocal rejects blacklisted repo")

# Test GitHub Code Review negative pair rejection
cr_neg = dict(cr_indented)
cr_neg["is_negative"] = True
assert map_codereview_to_bifocal(cr_neg, {"bad/repo"}) is None
print("[PASS] map_codereview_to_bifocal rejects is_negative==True")

# Test NextCoder
nc_sample = {
    "prompt": "Rewrite the given python program as per the following instruction.\nReplace recursion with iteration.\nWrite the entire code and no other text in the response.\n```python\ndef fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n```",
    "completion": "```python\ndef fact(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n```",
}
res_nc = map_nextcoder_to_bifocal(nc_sample, 0)
assert res_nc is not None
assert res_nc["instruction"] == "Replace recursion with iteration."
assert res_nc["target_patch"].startswith("diff --git a/solution.py b/solution.py")
assert res_nc["group"].startswith("nextcoder_seed_")
print("[PASS] map_nextcoder_to_bifocal")

print("[ALL ADAPTER TESTS PASSED EMPIRICALLY]")
