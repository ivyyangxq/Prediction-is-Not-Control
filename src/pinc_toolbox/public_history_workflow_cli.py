"""JSON and terminal driver for :class:`PublicHistoryWorkflow`."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .public_history_workflow import PublicHistoryWorkflow


_ACTIONS = frozenset({"inspect", "fit_reference", "build_region", "continue",
                      "test", "skip", "stop", "resume", "more_data"})
_NODES = frozenset({"C0", "CL", "CM", "CMIX"})


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _summary(snapshot, state):
    phase = snapshot.get("phase", "unknown")
    status = snapshot.get("status", "unknown")
    choices = snapshot.get("choices") or []
    initialization = snapshot.get("initialization") or {}
    reference = initialization.get("reference") or {}
    source = snapshot.get("source_workflow") or {}
    router = source.get("router") or {}
    return [
        f"status: {status}; phase: {phase}",
        f"initialization: {initialization.get('status', 'not_run')}; "
        f"N={initialization.get('N', 'n/a')}; T={initialization.get('T', 'n/a')}; "
        f"eta={initialization.get('eta_exact', initialization.get('selected_eta_exact', 'n/a'))}; "
        f"reason={initialization.get('reason', initialization.get('reasons', 'n/a'))}",
        f"reference theta/scale: {reference.get('candidate_theta_exact', 'n/a')} / "
        f"{reference.get('scale_exact', initialization.get('scale_exact', 'n/a'))}",
        f"initialization region: {(snapshot.get('prepared') or {}).get('readiness', {}).get('status', 'not_run')}; "
        f"source conclusion: {router.get('outcome', source.get('status', 'not_run'))}",
        f"choices: {', '.join(item.get('label', item.get('action', '')) for item in choices)}",
        f"state file: {state}",
    ]


def _result(snapshot, state, error=None):
    value = {"snapshot": snapshot, "summary": _summary(snapshot, state)}
    if error is not None:
        value["error"] = error
    return value


def _load(request, state):
    state = Path(state)
    if state.exists():
        stored = _read(state)
        if not isinstance(stored, dict) or stored.get("request") != request:
            raise ValueError("saved public-history request identity does not match")
        return PublicHistoryWorkflow.from_state(stored.get("workflow"), request)
    return PublicHistoryWorkflow(request)


def _persist(workflow, request, state):
    snapshot = workflow.snapshot()
    _atomic(state, {"request": request, "workflow": snapshot})
    return snapshot


def _apply(request, state, action, node=None, output=None):
    try:
        workflow = _load(request, state)
        before = workflow.snapshot()
        if action != "inspect":
            snapshot = workflow.step(action, node=node)
            snapshot = _persist(workflow, request, state)
        else:
            snapshot = before
        result = _result(snapshot, state)
    except Exception as error:
        result = _result(locals().get("before", {}), state,
                         {"type": type(error).__name__, "message": str(error),
                          "state_preserved": True})
    if output is not None:
        _atomic(output, result)
    return result


def _auto(request, state, output=None):
    try:
        workflow = _load(request, state)
        while True:
            snapshot = workflow.snapshot()
            # Auto mode never silently fits a reference or builds a region.
            if snapshot.get("phase") != "source" or snapshot.get("status") != "pending":
                result = _result(snapshot, state)
                break
            node = (snapshot.get("source_workflow") or {}).get("pending_node")
            if node is None:
                result = _result(snapshot, state)
                break
            workflow.step("continue", node=node)
            snapshot = _persist(workflow, request, state)
            result = _result(snapshot, state)
            if snapshot.get("status") == "resolved":
                break
    except Exception as error:
        result = _result(locals().get("snapshot", {}), state,
                         {"type": type(error).__name__, "message": str(error),
                          "state_preserved": True})
    if output is not None:
        _atomic(output, result)
    return result


def _print(result, verbose):
    if verbose:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("\n".join(result.get("summary", [])))
        if result.get("error"):
            print(f"error: {result['error'].get('message', 'workflow failed')}", file=sys.stderr)


def _interactive(request, state, output=None, verbose=False):
    while True:
        result = _apply(request, state, "inspect")
        _print(result, verbose)
        snapshot = result.get("snapshot", {})
        if result.get("error") or snapshot.get("status") == "resolved":
            if output is not None:
                _atomic(output, result)
            return result
        choices = snapshot.get("choices") or []
        if not choices:
            raise RuntimeError("workflow returned no permitted choices")
        print("Choose one action:")
        for index, choice in enumerate(choices, 1):
            print(f"  {index}: {choice.get('label', choice.get('action'))}")
        try:
            raw = input("Choice number: ").strip()
        except EOFError:
            result = _apply(request, state, "stop")
            _print(result, verbose)
            if output is not None:
                _atomic(output, result)
            return result
        if not raw.isdigit() or not 1 <= int(raw) <= len(choices):
            print("Unknown choice; enter one of the displayed numbers.", file=sys.stderr)
            continue
        choice = choices[int(raw) - 1]
        action, node = choice.get("action"), choice.get("node")
        result = _apply(request, state, action, node=node)
        _print(result, verbose)
        if result.get("error") or action in {"stop", "more_data"}:
            if output is not None:
                _atomic(output, result)
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--action", choices=sorted(_ACTIONS))
    group.add_argument("--auto", action="store_true")
    parser.add_argument("--node", choices=sorted(_NODES))
    args = parser.parse_args(argv)
    if args.state.resolve() == args.input.resolve():
        parser.error("--state must be distinct from --input")
    if args.output is not None and args.output.resolve() in {args.input.resolve(), args.state.resolve()}:
        parser.error("--output must be distinct from --input and --state")
    if args.output is not None and args.output.exists():
        parser.error("--output must be fresh")
    if args.node is not None and (args.action not in {"continue", "test", "skip"}):
        parser.error("--node requires a source action")
    request = _read(args.input)
    if args.auto:
        result = _auto(request, args.state, args.output)
    elif args.action is None:
        result = _interactive(request, args.state, args.output, args.verbose)
    else:
        result = _apply(request, args.state, args.action, args.node, args.output)
    if args.action is not None or args.auto:
        _print(result, args.verbose)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
