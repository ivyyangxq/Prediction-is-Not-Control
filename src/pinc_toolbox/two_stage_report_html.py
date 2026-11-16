"""Self-contained HTML rendering for a saved two-stage report.

This module only formats the supplied report. It does not run the detector,
source evaluator, or any statistical procedure.
"""
from __future__ import annotations

import html
import json
from collections.abc import Mapping


def _esc(value) -> str:
    if value is None:
        return "<span class=\"unavailable\">Not provided</span>"
    if isinstance(value, bool):
        value = "Yes" if value else "No"
    elif isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return html.escape(str(value), quote=True)


def _number(value, kind: str) -> str:
    if value is None:
        label = "Not established" if kind == "type_i" else "Not estimated"
        return f'<span class="unavailable">{label}</span>'
    return _esc(value)


def _error_rows(errors, *, include_targets=False) -> str:
    if not isinstance(errors, Mapping):
        return '<tr><td colspan="6">No error-rate record provided</td></tr>'
    rows = []
    for family, measures in errors.items():
        if not isinstance(measures, Mapping):
            continue
        for kind in ("type_i", "type_ii"):
            item = measures.get(kind)
            if not isinstance(item, Mapping):
                continue
            rows.append(
                "<tr>"
                f"<td>{_esc(family)}</td><td>{_esc(kind)}</td>"
                + (f"<td>{_esc(item.get('target'))}</td>" if include_targets else "")
                +
                f"<td>{_esc(item.get('conditioning'))}</td>"
                f"<td>{_number(item.get('target_budget'), kind)}</td>"
                f"<td>{_number(item.get('proved_bound'), kind)}</td>"
                f"<td>{_number(item.get('empirical'), kind)}</td>"
                "</tr>"
            )
    return "".join(rows) or '<tr><td colspan="6">No error-rate record provided</td></tr>'


def render_html(report) -> str:
    """Render a complete, dependency-free HTML document from a saved report."""
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    stage1 = report.get("stage1") or {}
    stage2 = report.get("stage2") or {}
    labels = report.get("source_labels") or {}
    disclosure = report.get("semantic_disclosure", "")
    context = report.get("context") or {}
    conditional = context.get("route") == "conditional_history_saved_report"
    source_probability = "Unavailable"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    errors_table = _error_rows(report.get("errors"))
    if conditional:
        errors_table = _error_rows(report.get("errors"), include_targets=True)
        shared = (report.get("errors") or {}).get("joint", {}).get("resolved_assertion_error", {})
        errors_table += (
            '<tr><td>joint: resolved assertion (shared)</td><td>shared resolved error</td>'
            f'<td>{_esc(shared.get("target"))}</td>'
            f'<td>{_esc(shared.get("conditioning"))}</td>'
            f'<td>{_number(shared.get("target_budget"), "type_i")}</td>'
            f'<td>{_number(shared.get("proved_bound"), "type_i")}</td>'
            f'<td>{_number(shared.get("empirical"), "type_i")}</td></tr>')
    capacity = ""
    if "calibration_capacity_explanation" in stage1:
        capacity = (
            '<p>Failed CAL slots: ' + _esc(stage1.get("cal_failed_count")) +
            ' / ' + _esc(stage1.get("cal_total_B")) +
            '; p-value floor from failed slots: ' + _esc(stage1.get("calibration_failure_p_floor")) +
            '; rejection allowed by this floor alone: ' + _esc(stage1.get("calibration_allows_rejection")) +
            '.</p><p>' + _esc(stage1["calibration_capacity_explanation"]) + '</p>')
    if not conditional:
        return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two-Stage Diagnosis Report</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#202124}
h1,h2{line-height:1.2} .unavailable{color:#7a4b00;font-weight:600}
table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #bbb;padding:.45rem;text-align:left;vertical-align:top}
th{background:#f2f4f7} pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1rem}
details{border-top:1px solid #bbb;margin-top:1.5rem;padding-top:.8rem} summary{cursor:pointer;font-weight:650}
</style>
</head>
<body>
<main>
<h1>Two-Stage Diagnosis Report</h1>
<section aria-labelledby="conclusions"><h2 id="conclusions">Conclusions</h2>
<p>Stage 1: <strong>""" + _esc(stage1.get("status")) + """</strong></p>
""" + capacity + """
<p>Stage 2: <strong>""" + _esc(stage2.get("outcome")) + """</strong>;
""" + _esc(labels.get("latent", "latent")) + """: """ + _esc(stage2.get("latent")) + """; """ + _esc(labels.get("model", "model")) + """: """ + _esc(stage2.get("model")) + """;
mixed: """ + _esc(stage2.get("mixed")) + """</p>
<p>""" + _esc(disclosure) + """</p>
<p>Source probability: <strong>""" + source_probability + """</strong>. Source necessity is relative to the declared candidate classes; it is not the probability that a source occurred.</p>
</section>
<details><summary>Details</summary>
<section aria-labelledby="errors"><h2 id="errors">Type I / Type II Definitions and Values</h2>
<p>Detection Type I: an alarm when no change occurred; detection Type II: no completed alarm by the deadline under a specified change.
Source Type I: reporting a component as necessary when it did not change; source Type II: failing to complete a necessity report under a specified component change.
Stage 2 is conditional on a Stage 1 alarm; mixed means that both necessity reports hold.</p>
<p>The table distinguishes prespecified budgets, theoretical bounds under their assumptions, and estimates from matched experiments. A theoretical bound does not verify the assumptions for this input.
An exact-set certificate does not give a floating-point statistical radius machine-exact coverage. Not established or not estimated does not mean zero.</p>
<table><thead><tr><th>Test</th><th>Type</th><th>Conditioning</th><th>Target budget</th><th>Proved upper bound</th><th>Matched experiment</th></tr></thead>
<tbody>""" + errors_table + """</tbody></table>
</section>
<h2>Complete Saved Report</h2><pre>""" + html.escape(encoded, quote=True) + """</pre></details>
</main>
</body>
</html>
"""
    if conditional:
        conclusion = (
            '<p>Stage 2: <strong>' + _esc(stage2.get("outcome")) + '</strong>; '
            'model: ' + _esc(stage2.get("model")) + '; latent: ' + _esc(stage2.get("latent")) + '</p>'
            '<p>Retained candidates: <strong>' + _esc(stage2.get("retained")) + '</strong></p>'
            '<p>Mixed status: <strong>' + _esc(stage2.get("mixed_status")) + '</strong>; '
            'mixed Boolean (joint_presence_asserted_only): <strong>' + _esc(stage2.get("mixed")) + '</strong>. '
            'False does not mean mixed absent.</p>'
            '<p>Error scope: ' + _esc(context.get("error_scope")) + '</p>')
        error_explanation = (
            '<p>Source Type I targets only false presence assertions; source Type II is not estimated. '
            'Unresolved/withheld is not a presence assertion. The joint shared resolved-assertion field specifies a shared error target, '
            'including false presence or absence assertions but not unresolved misses. Consult the error scope for whether an applicable bound has been established.</p>'
            '<p>Error scope: ' + _esc(context.get("error_scope")) + '</p>'
            '<p>Execution scope: ' + _esc(context.get("execution_scope")) + '</p>')
        probability_explanation = 'Source probability is unavailable. A p-value is a rank including the observed value, not a posterior probability.'
        table_header = '<tr><th>Test</th><th>Type</th><th>Error target</th><th>Conditioning</th><th>Target budget</th><th>Proved upper bound</th><th>Matched experiment</th></tr>'
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two-Stage Diagnosis Report</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#202124}
h1,h2{line-height:1.2} .unavailable{color:#7a4b00;font-weight:600}
table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #bbb;padding:.45rem;text-align:left;vertical-align:top}
th{background:#f2f4f7} pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1rem}
details{border-top:1px solid #bbb;margin-top:1.5rem;padding-top:.8rem} summary{cursor:pointer;font-weight:650}
</style>
</head>
<body>
<main>
<h1>Two-Stage Diagnosis Report</h1>
<section aria-labelledby="conclusions"><h2 id="conclusions">Conclusions</h2>
<p>Stage 1: <strong>""" + _esc(stage1.get("status")) + """</strong></p>
""" + capacity + conclusion + """
<p>""" + _esc(disclosure) + """</p>
<p>""" + probability_explanation + """</p>
</section>
<details><summary>Details</summary>
<section aria-labelledby="errors"><h2 id="errors">Type I / Type II Definitions and Values</h2>
""" + error_explanation + """
<p>The table distinguishes prespecified budgets, theoretical bounds under their assumptions, and estimates from matched experiments. A theoretical bound does not verify the assumptions for this input.
An exact-set certificate does not give a floating-point statistical radius machine-exact coverage. Not established or not estimated does not mean zero.</p>
<table><thead>""" + table_header + """</thead>
<tbody>""" + errors_table + """</tbody></table>
</section>
<h2>Complete Saved Report</h2><pre>""" + html.escape(encoded, quote=True) + """</pre></details>
</main>
</body>
</html>
"""


__all__ = ["render_html"]
