"""AST-only target validation with auditable, scoped SyntaxWarning diagnostics."""

import ast
import warnings

SYNTAX_DIAGNOSTICS_POLICY = "ast_syntax_warnings_recorded_not_rejected_v1"


def inspect_python_target(code, *, filename):
    """Never execute or rewrite source; preserve syntax errors and other warnings."""
    parse_error = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            ast.parse(code, filename=filename)
        except (SyntaxError, ValueError, RecursionError) as error:
            parse_error = error
    diagnostics = []
    for warning in captured:
        if issubclass(warning.category, SyntaxWarning):
            message = str(warning.message)
            diagnostics.append(
                {
                    "category": warning.category.__name__,
                    "kind": "invalid_escape_sequence"
                    if "invalid escape sequence" in message
                    else "other_syntax_warning",
                    "message": message,
                    "filename": warning.filename,
                    "line": warning.lineno,
                }
            )
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    if parse_error is not None:
        raise ValueError("invalid_target_python_syntax") from parse_error
    return diagnostics


def empty_syntax_warning_summary():
    return {
        "parsed_targets_with_warnings": 0,
        "parsed_warning_events": 0,
        "parsed_by_kind": {},
        "accepted_targets_with_warnings": 0,
        "accepted_warning_events": 0,
        "accepted_by_kind": {},
    }


def count_syntax_warnings(summary, diagnostics, *, accepted=False):
    """Parsed totals include later-rejected examples; admitted totals are separate."""
    if not diagnostics:
        return
    prefix = "accepted" if accepted else "parsed"
    summary[prefix + "_targets_with_warnings"] += 1
    summary[prefix + "_warning_events"] += len(diagnostics)
    kinds = summary[prefix + "_by_kind"]
    for item in diagnostics:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1


def summarize_admitted_warning_records(records):
    """Final-cohort counts, separate from parsing/admission totals before token caps."""
    summary = empty_syntax_warning_summary()
    for row in records:
        count_syntax_warnings(summary, row.get("target_syntax_warnings", []), accepted=True)
    return {
        "examples": len(records),
        "examples_with_syntax_warnings": summary["accepted_targets_with_warnings"],
        "warning_events": summary["accepted_warning_events"],
        "by_kind": summary["accepted_by_kind"],
    }
