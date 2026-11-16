"""Terminal and JSON driver for the one-node-at-a-time source workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


_ACTIONS = frozenset({'inspect', 'continue', 'test', 'skip', 'stop', 'resume', 'more_data'})
_NODES = frozenset({'C0', 'CL', 'CM', 'CMIX'})


def _read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=target.parent,
                                         prefix=f'.{target.name}.', suffix='.tmp',
                                         delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _request_parts(request):
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object')
    if 'raw_history' in request:
        if request.get('source') is not None or 'readiness' in request:
            raise ValueError('raw_history requests must omit source and external readiness')
        context = request.get('context', {})
        if not isinstance(context, dict):
            raise ValueError('context must be a JSON object')
        return (context, None, request.get('goal', 'full'), request.get('available'),
                request.get('readiness'), True, request['raw_history'], request.get('window_index'))
    required = ('context', 'goal', 'available', 'readiness')
    missing = [name for name in required if name not in request]
    if missing:
        raise ValueError(f'request missing required fields: {missing}')
    if not isinstance(request['context'], dict):
        raise ValueError('context must be a JSON object')
    source = request.get('source')
    if source is not None and not isinstance(source, dict):
        raise ValueError('source must be a JSON object when supplied')
    return request['context'], source, request['goal'], request['available'], request['readiness'], False, None, None


def _identity(request):
    return request


def _readiness_blocks(readiness):
    if isinstance(readiness, dict):
        status = readiness.get('status')
    else:
        status = readiness
    return status not in {'ready', 'supported', 'available'}


class _UnavailableEvaluator:
    snapshot = {'status': 'unavailable', 'reason': 'source readiness blocks evaluation'}

    def __call__(self, class_name):
        raise RuntimeError('source evaluation is unavailable until readiness permits it')


def _load_or_create(request, state_path):
    from .source_workflow import SourceWorkflow

    context, source, goal, available, readiness, raw_mode, raw_request, window_index = _request_parts(request)
    identity = _identity(request)
    state_file = Path(state_path)
    workflow_state = None
    if state_file.exists():
        stored = _read_json(state_file)
        if not isinstance(stored, dict) or stored.get('identity') != identity:
            raise ValueError('state identity does not match the complete current request')
        workflow_state = stored.get('workflow')
        if not isinstance(workflow_state, dict):
            raise ValueError('state file has no workflow snapshot')
    preparation = None
    if raw_mode and workflow_state is None:
        from .raw_history_source_workflow import prepare_raw_history
        if window_index is None:
            raise ValueError('raw_history request requires window_index')
        preparation = prepare_raw_history(raw_request, window_index=window_index)
        readiness = preparation.get('readiness')
        source = preparation.get('source')
    elif raw_mode and workflow_state is not None:
        stored_preparation = stored.get('preparation')
        if not isinstance(stored_preparation, dict):
            raise ValueError('raw_history state has no prepared region')
        preparation = stored_preparation
        readiness = preparation.get('readiness')
        source = preparation.get('source')
    if _readiness_blocks(readiness):
        evaluator = _UnavailableEvaluator()
    else:
        from .certified_source_workflow_backend import CertifiedSourceEvaluator
        evaluator = CertifiedSourceEvaluator(source)
    if workflow_state is not None:
        workflow = SourceWorkflow.from_state(workflow_state, evaluator, context=identity)
    else:
        workflow = SourceWorkflow(evaluator, identity, goal=goal,
                                  available=available, readiness=readiness)
    return workflow, evaluator, identity, preparation


def _persist(workflow, evaluator, identity, state_path, preparation=None):
    snapshot = workflow.snapshot()
    evaluator_snapshot = evaluator.snapshot() if callable(evaluator.snapshot) else evaluator.snapshot
    envelope = {'identity': identity, 'workflow': snapshot, 'evaluator': evaluator_snapshot}
    if preparation is not None:
        envelope['preparation'] = preparation
    _atomic_json(state_path, envelope)
    return snapshot


def _summary(snapshot, state_path=None):
    statuses = snapshot.get('statuses') or {}
    tested = statuses.get('tested') or {}
    inherited = statuses.get('inherited') or {}
    excluded = sorted(node for node, status in tested.items() if status == 'excluded')
    compatible = sorted(node for node, status in tested.items() if status == 'compatible')
    unresolved = sorted(node for node, status in tested.items() if status == 'unresolved')
    unknown = sorted(set(statuses.get('not_run') or []) | set(statuses.get('unavailable') or []))
    skipped = sorted(statuses.get('user_skipped') or [])
    router = snapshot.get('router') or {}
    outcome = router.get('outcome') or snapshot.get('status') or 'unknown'
    conclusion = {
        'ambiguous': 'both pure explanations remain possible',
        'both_required': 'joint explanation is required after both pure explanations were excluded',
        'family_inadequate': 'declared source family is inadequate',
        'maintained_family_inadequate': 'declared source family is inadequate',
        'model_required': 'model change is required',
        'latent_required': 'latent change is required',
        'no_change_not_excluded': 'no-change source remains possible',
        'model_change_not_established': 'model change is not established',
        'unresolved': 'source conclusion remains unresolved',
    }.get(outcome, str(outcome))
    pending = snapshot.get('pending_node') or 'none'
    path = snapshot.get('path') or []
    last = path[-1] if path else None
    last_step = 'none' if not last else f"{last.get('action')} {last.get('node') or ''}".strip()
    return [
        f"status: {snapshot.get('status', 'unknown')}; conclusion: {conclusion}",
        f"compatible (not excluded, not proof): {', '.join(compatible) if compatible else 'none'}; "
        f"excluded (evidence): {', '.join(excluded) if excluded else 'none'}",
        f"unresolved: {', '.join(unresolved) if unresolved else 'none'}; "
        f"unknown: {', '.join(unknown) if unknown else 'none'}; "
        f"skipped: {', '.join(skipped) if skipped else 'none'}",
        f"inherited compatible: {', '.join(sorted(inherited)) if inherited else 'none'} "
        f"; pending recommendation: {pending}; last step: {last_step}",
        f"state file: {state_path if state_path is not None else '<unspecified>'}",
    ]


def _result(snapshot, *, error=None, preparation=None, state_path=None):
    output = {'snapshot': snapshot, 'summary': _summary(snapshot, state_path)}
    if preparation is not None:
        output['preparation'] = preparation
    if error is not None:
        output['error'] = error
    return output


def _write_output(path, value):
    if path is None:
        return
    _atomic_json(path, value)


def _print_interactive(result, verbose=False):
    if verbose:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print('\n'.join(result.get('summary', [])))
        if result.get('error'):
            print(f"error: {result['error'].get('message', 'workflow failed')}", file=sys.stderr)


def _apply(request, state_path, action, node=None, output_path=None):
    try:
        workflow, evaluator, identity, preparation = _load_or_create(request, state_path)
        before = workflow.snapshot()
    except Exception as error:
        result = _result({}, error={'type': type(error).__name__, 'message': str(error),
                                    'state_preserved': True}, state_path=state_path)
        _write_output(output_path, result)
        return result
    if action not in _ACTIONS:
        raise ValueError(f'unknown action {action!r}; choose one of {sorted(_ACTIONS)}')
    if node is not None and node not in _NODES:
        raise ValueError(f'unknown node {node!r}; choose one of {sorted(_NODES)}')
    try:
        if action != 'inspect':
            workflow.step(action, node=node, context=identity)
        if action == 'inspect':
            if preparation is not None and not Path(state_path).exists():
                snapshot = _persist(workflow, evaluator, identity, state_path, preparation)
            else:
                snapshot = before
        else:
            try:
                snapshot = _persist(workflow, evaluator, identity, state_path, preparation)
            except Exception as error:
                result = _result(before, error={'type': type(error).__name__, 'message': str(error),
                                                'state_preserved': False}, state_path=state_path)
                _write_output(output_path, result)
                return result
        result = _result(snapshot, preparation=preparation, state_path=state_path)
    except Exception as error:
        # The workflow object is discarded and the prior state file is left intact.
        result = _result(before, error={'type': type(error).__name__, 'message': str(error),
                                        'state_preserved': True}, state_path=state_path)
    _write_output(output_path, result)
    return result


def _interactive(request, state_path, output_path=None, verbose=False):
    try:
        workflow, evaluator, identity, preparation = _load_or_create(request, state_path)
    except Exception as error:
        result = _result({}, error={'type': type(error).__name__, 'message': str(error),
                                    'state_preserved': True}, state_path=state_path)
        _write_output(output_path, result)
        return result
    while True:
        result = _result(workflow.snapshot(), preparation=preparation, state_path=state_path)
        if preparation is not None and not Path(state_path).exists():
            result = _result(_persist(workflow, evaluator, identity, state_path, preparation),
                             preparation=preparation, state_path=state_path)
        _print_interactive(result, verbose=verbose)
        snapshot = result['snapshot']
        if snapshot.get('status') == 'resolved':
            _write_output(output_path, result)
            return result
        choices = snapshot.get('choices') or []
        choice_nodes = {}
        choice_entries = []
        for item in choices:
            action_name = item.get('action') if isinstance(item, dict) else item
            node_name = item.get('node') if isinstance(item, dict) else None
            label = item.get('label') if isinstance(item, dict) else str(action_name)
            if action_name in _ACTIONS:
                choice_entries.append((action_name, node_name, label or action_name))
                choice_nodes.setdefault(action_name, []).append(node_name)
        if not any(action == 'stop' for action, _, _ in choice_entries):
            choice_entries.append(('stop', None, 'Stop and save'))
        permitted = list(dict.fromkeys(action for action, _, _ in choice_entries))
        print('Choose one action:')
        for choice_index, (choice_action, choice_node, choice_label) in enumerate(choice_entries, 1):
            suffix = '' if choice_node is None else f' [{choice_node}]'
            print(f'  {choice_index}: {choice_label}{suffix}')
        try:
            raw_action = input('choice> ').strip()
        except EOFError:
            raw_action = 'stop'
        node = None
        if raw_action.isdigit() and 1 <= int(raw_action) <= len(choice_entries):
            action, node, _ = choice_entries[int(raw_action) - 1]
        else:
            action = raw_action
        if action not in permitted:
            print(f'Unknown action {action!r}; choose one of {permitted}', file=sys.stderr)
            continue
        if action in {'continue', 'test', 'skip'}:
            try:
                allowed_nodes = [item for item in choice_nodes.get(action, []) if item is not None]
                if node is not None:
                    pass
                elif len(allowed_nodes) == 1:
                    node = allowed_nodes[0]
                elif allowed_nodes:
                    node = input('node (' + '/'.join(allowed_nodes) + ', blank for core choice)> ').strip() or None
                else:
                    node = input('node (C0/CL/CM/CMIX, blank for core choice)> ').strip() or None
            except EOFError:
                action, node = 'stop', None
        allowed_nodes = [item for item in choice_nodes.get(action, []) if item is not None]
        if allowed_nodes and node not in allowed_nodes:
            print(f'Node {node!r} is not permitted for action {action!r}; choose {allowed_nodes}', file=sys.stderr)
            continue
        before = workflow.snapshot()
        try:
            workflow.step(action, node=node, context=identity)
            result = _result(_persist(workflow, evaluator, identity, state_path, preparation),
                             preparation=preparation, state_path=state_path)
        except Exception as error:
            result = _result(before, error={'type': type(error).__name__, 'message': str(error),
                                            'state_preserved': True}, state_path=state_path)
        _print_interactive(result, verbose=verbose)
        if result.get('error') or action in {'stop', 'more_data'} or result['snapshot'].get('status') == 'resolved':
            _write_output(output_path, result)
            return result


def _auto(request, state_path, output_path=None):
    try:
        workflow, evaluator, identity, preparation = _load_or_create(request, state_path)
        while True:
            snapshot = workflow.snapshot()
            if snapshot.get('status') != 'pending' or snapshot.get('pending_node') is None:
                result = _result(snapshot, preparation=preparation, state_path=state_path)
                if preparation is not None and not Path(state_path).exists():
                    result = _result(_persist(workflow, evaluator, identity, state_path, preparation),
                                     preparation=preparation, state_path=state_path)
                _write_output(output_path, result)
                return result
            node = snapshot['pending_node']
            workflow.step('continue', node=node, context=identity)
            result = _result(_persist(workflow, evaluator, identity, state_path, preparation),
                             preparation=preparation, state_path=state_path)
    except Exception as error:
        result = _result({}, error={'type': type(error).__name__, 'message': str(error),
                                    'state_preserved': True}, state_path=state_path)
        _write_output(output_path, result)
        return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verbose', action='store_true')
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--action', choices=sorted(_ACTIONS))
    action_group.add_argument('--auto', action='store_true')
    parser.add_argument('--node', choices=sorted(_NODES))
    args = parser.parse_args(argv)
    if args.state.resolve() == args.input.resolve():
        parser.error('--state must be distinct from --input')
    if args.output is not None and args.output.resolve() in {args.input.resolve(), args.state.resolve()}:
        parser.error('--output must be a fresh path distinct from --input and --state')
    if args.output is not None and args.output.exists():
        parser.error('--output must be a fresh path that does not already exist')
    if args.node is not None and (args.action is None or args.action not in {'continue', 'test', 'skip'}):
        parser.error('--node requires --action continue, test, or skip')
    request = _read_json(args.input)
    if args.auto:
        result = _auto(request, args.state, args.output)
        _print_interactive(result, verbose=args.verbose)
    elif args.action is None:
        result = _interactive(request, args.state, args.output, verbose=args.verbose)
    else:
        result = _apply(request, args.state, args.action, node=args.node, output_path=args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if 'error' not in result else 1


if __name__ == '__main__':
    raise SystemExit(main())
