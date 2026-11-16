"""Lazy source diagnosis from caller-supplied structural evidence.

The router never computes a confidence region or interprets a detector score.
It only orders and records evidence for the nested classes C0 ⊂ CL, CM ⊂ CMIX.
"""

import json
import math


_CLASSES = ("C0", "CL", "CM", "CMIX")
_STATUSES = {"compatible", "excluded", "unresolved", "unavailable"}


def _json_finite(value):
    """Validate that *value* can be emitted as strict JSON."""
    json.dumps(value, allow_nan=False)
    return value


def _evidence(class_name, raw):
    if not isinstance(raw, dict):
        raise TypeError("evaluate(class_name) must return a mapping")
    item = dict(raw)
    status = item.get("status")
    if status not in _STATUSES:
        raise ValueError("%s evidence has invalid status %r" % (class_name, status))
    if status in {"compatible", "excluded"}:
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                "%s %s evidence requires an upstream description"
                % (class_name, status)
            )
    item.setdefault("provenance", None)
    item.setdefault("reason", None)
    _json_finite(item)
    return item


def diagnose_sources(evaluate, *, alarm=None, available=None, goal="full"):
    """Route source checks and classify the retained explanation family.

    ``evaluate`` is called at most once per class and receives one of ``C0``,
    ``CL``, ``CM`` or ``CMIX``.  It must return a mapping with a ``status``
    and, for compatible/excluded evidence, an actual upstream ``description``.
    Callback exceptions intentionally propagate.  ``available`` optionally
    names classes that may be queried; omitted classes are marked unavailable
    without invoking the callback.
    """
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    if goal not in {"full", "model_required"}:
        raise ValueError("goal must be 'full' or 'model_required'")
    allowed = None if available is None else set(available)
    unknown = set() if allowed is None else allowed.difference(_CLASSES)
    if unknown:
        raise ValueError("unknown source classes: %s" % sorted(unknown))
    evidence = {name: {"status": "not_run"} for name in _CLASSES}
    trace = {"visited": [], "inherited": [], "not_run": list(_CLASSES),
             "skipped_unavailable": []}
    cache = {}
    callback_count = 0

    def set_item(name, item, *, inherited_from=None):
        evidence[name] = item
        if name in trace["not_run"]:
            trace["not_run"].remove(name)
        if inherited_from is None:
            trace["visited"].append(name)
        else:
            trace["inherited"].append({"class_name": name, "from": inherited_from,
                                       "relation": "upper_bound_from_subset"})

    def visit(name):
        nonlocal callback_count
        if name in cache:
            return cache[name]
        if allowed is not None and name not in allowed:
            item = {"status": "unavailable", "provenance": None,
                    "reason": "class not declared available"}
            cache[name] = item
            evidence[name] = item
            trace["not_run"].remove(name)
            trace["skipped_unavailable"].append(name)
            return item
        else:
            callback_count += 1
            item = _evidence(name, evaluate(name))
        cache[name] = item
        set_item(name, item)
        return item

    def inherit(name, source):
        parent = dict(evidence[source])
        parent["inherited_from"] = source
        parent["upstream_provenance"] = parent.get("provenance")
        parent["upper_bound_from_subset"] = True
        cache[name] = parent
        set_item(name, parent, inherited_from=source)

    c0 = visit("C0")
    if c0["status"] == "compatible":
        for name in ("CL", "CM", "CMIX"):
            inherit(name, "C0")
        outcome = "no_change_not_excluded"
        next_check = None
    else:
        cl = visit("CL")
        if cl["status"] == "compatible":
            inherit("CMIX", "CL")
            if goal == "model_required":
                outcome = "model_change_not_established"
                next_check = None
                cm = None
            else:
                cm = visit("CM")
        else:
            cm = visit("CM")
        if cm is not None and cm["status"] == "compatible" and evidence["CMIX"]["status"] == "not_run":
            inherit("CMIX", "CM")
        if cm is None:
            pass
        elif cl["status"] == "compatible" and cm["status"] == "compatible":
            outcome = "ambiguous"
            next_check = None
        elif cl["status"] == "compatible" and cm["status"] == "excluded":
            outcome = "latent_required"
            next_check = None
        elif cl["status"] == "excluded" and cm["status"] == "compatible":
            outcome = "model_required"
            next_check = None
        elif cl["status"] == "excluded" and cm["status"] == "excluded":
            mix = visit("CMIX")
            if mix["status"] == "compatible":
                outcome = "both_required"
            elif mix["status"] == "excluded":
                outcome = "maintained_family_inadequate"
            else:
                outcome = "unresolved"
            next_check = None if mix["status"] in {"compatible", "excluded"} else "CMIX"
        elif cl["status"] == "excluded" and cm["status"] != "excluded":
            mix = visit("CMIX")
            if mix["status"] == "compatible":
                outcome = "model_required"
                next_check = None
            elif mix["status"] == "excluded":
                outcome = "maintained_family_inadequate"
                next_check = None
            else:
                outcome = "unresolved"
                next_check = "CMIX"
        elif cm["status"] == "excluded" and cl["status"] != "excluded":
            mix = visit("CMIX")
            if mix["status"] == "compatible":
                outcome = "latent_required"
                next_check = None
            elif mix["status"] == "excluded":
                outcome = "maintained_family_inadequate"
                next_check = None
            else:
                outcome = "unresolved"
                next_check = "CMIX"
        else:
            # An unresolved/unavailable pure result does not become exclusion.
            # CMIX can still supply the remaining maintained-family explanation.
            mix = visit("CMIX")
            if mix["status"] == "excluded":
                outcome = "maintained_family_inadequate"
                next_check = None
            elif mix["status"] == "compatible":
                outcome = "unresolved"
                next_check = None
            else:
                outcome = "unresolved"
                next_check = "CMIX"

    result = {
        "outcome": outcome,
        "evidence": evidence,
        "trace": {
            "visited": trace["visited"],
            "inherited": trace["inherited"],
            "not_run": trace["not_run"],
            "skipped_unavailable": trace["skipped_unavailable"],
            "next_useful_check": next_check,
            "need_new_information": next_check is not None or outcome in {"unresolved", "ambiguous"},
        },
        "callbacks": callback_count,
        "alarm": alarm,
    }
    return _json_finite(result)
