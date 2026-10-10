"""Dataset parser diagnostics do not change targets or suppress runtime warnings."""

import warnings

import pytest

from bifocalgemma.data.curation import CodeOverlapIndex, curate_paired_sources
from bifocalgemma.data.python_quality import (
    count_syntax_warnings,
    empty_syntax_warning_summary,
    inspect_python_target,
    summarize_admitted_warning_records,
)
from tests.test_curation import raw


def test_warning_has_provenance_without_stderr_or_source_mutation(capsys):
    code = 'pattern = "' + chr(92) + 'd+"\n'
    original = code
    filters = list(warnings.filters)
    diagnostics = inspect_python_target(code, filename="commitpackft[12]:pattern.py")
    assert code == original and warnings.filters == filters
    assert capsys.readouterr().err == ""
    assert len(diagnostics) == 1
    assert diagnostics[0]["kind"] == "invalid_escape_sequence"
    assert diagnostics[0]["category"] == "SyntaxWarning"
    assert diagnostics[0]["filename"] == "commitpackft[12]:pattern.py"
    assert diagnostics[0]["line"] == 1
    assert inspect_python_target('pattern = r"' + chr(92) + 'd+"\n', filename="raw.py") == []
    assert (
        inspect_python_target('raise RuntimeError("must never execute")', filename="target.py")
        == []
    )


def test_errors_are_still_rejected_and_other_warnings_still_visible(monkeypatch):
    with pytest.raises(ValueError, match="invalid_target_python_syntax"):
        inspect_python_target("def invalid(", filename="bad.py")
    import bifocalgemma.data.python_quality as quality

    def parser(*args, **kwargs):
        warnings.warn("keep this runtime warning", UserWarning)
        raise SyntaxError("bad source")

    monkeypatch.setattr(quality.ast, "parse", parser)
    with pytest.warns(UserWarning, match="keep this runtime warning"):
        with pytest.raises(ValueError, match="invalid_target_python_syntax"):
            inspect_python_target("x", filename="bad.py")


def test_curated_warning_target_retained_and_summary_distinguishes_duplicate(capsys):
    item = raw(8)
    item["new_contents"] = 'pattern = "' + chr(92) + 'd+"\n'
    original = item["new_contents"]
    records, report = curate_paired_sources(
        [("editpackft", [item, dict(item)])],
        CodeOverlapIndex(),
        blocked_repos=set(),
    )
    assert len(records) == 1 and records[0]["after"] == original
    assert records[0]["quality_flags"] == ["target_has_syntax_warning"]
    assert records[0]["target_syntax_warnings"][0]["filename"] == "editpackft[0]:a.py"
    summary = report["syntax_warning_summary"]["editpackft"]
    assert summary["parsed_targets_with_warnings"] == 2
    assert summary["accepted_targets_with_warnings"] == 1
    assert capsys.readouterr().err == ""
    progress_summary = empty_syntax_warning_summary()
    count_syntax_warnings(progress_summary, records[0]["target_syntax_warnings"])
    count_syntax_warnings(progress_summary, records[0]["target_syntax_warnings"], accepted=True)
    assert progress_summary["parsed_by_kind"] == {"invalid_escape_sequence": 1}
    assert progress_summary["accepted_by_kind"] == {"invalid_escape_sequence": 1}


def test_final_cohort_summary_does_not_include_filtered_examples():
    diagnostics = inspect_python_target('pattern = "' + chr(92) + 'd+"', filename="target.py")
    summary = summarize_admitted_warning_records(
        [{"target_syntax_warnings": diagnostics}, {"target_syntax_warnings": []}]
    )
    assert summary == {
        "examples": 2,
        "examples_with_syntax_warnings": 1,
        "warning_events": 1,
        "by_kind": {"invalid_escape_sequence": 1},
    }
