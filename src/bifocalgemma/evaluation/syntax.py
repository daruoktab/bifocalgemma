"""Static Python syntax of changed files, independent of functional correctness."""

import ast

from bifocalgemma.data.patches import apply_patch_to_snapshot, extract_unified_diff


def patch_syntax(snapshot, completion, *, applied):
    if not applied:
        return {
            "syntax_valid": None,
            "syntax_checked_files": 0,
            "syntax_note": "Patch did not apply.",
        }
    fixed = apply_patch_to_snapshot(snapshot, extract_unified_diff(completion))
    before = {f.path: f.content for f in snapshot.files}
    changed = [f for f in fixed.files if f.path.endswith(".py") and before.get(f.path) != f.content]
    if not changed:
        return {
            "syntax_valid": None,
            "syntax_checked_files": 0,
            "syntax_note": "No changed Python files.",
        }
    for file in changed:
        try:
            ast.parse(file.content, filename=file.path)
        except (SyntaxError, ValueError):
            return {"syntax_valid": False, "syntax_checked_files": len(changed)}
    return {"syntax_valid": True, "syntax_checked_files": len(changed)}
