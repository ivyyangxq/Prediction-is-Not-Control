"""JSON interface to the same PanelSession used by GQ_tool.ipynb."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .product import PanelSession, prepare_panel, demo_data, demo_settings, clean


def execute(request, *, directory=None):
    base = Path(directory or '.').resolve()
    def path(value):
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else base / candidate
    if request.get('demo') is True:
        panel, macro, mapping = demo_data()
        history = prepare_panel(panel, macro, mapping, 'macro_W', 'publication')
        settings = demo_settings()
    else:
        panel = pd.read_csv(path(request['panel_csv']))
        macro = pd.read_csv(path(request['macro_csv']))
        history = prepare_panel(panel, macro, request['mapping'],
                                request['macro_value'], request['macro_release'],
                                macro_unit=request.get('macro_unit'))
        settings = request.get('settings')
    if 'resume_report' in request:
        saved = request['resume_report']
        if isinstance(saved, str):
            saved = json.loads(path(saved).read_text(encoding='utf-8'))
        if saved.get('status') == 'ok' and 'report' in saved:
            saved = saved['report']
        if settings is not None and settings != saved['configuration']:
            raise ValueError('settings differ from resumed report; start a new session')
        session = PanelSession.from_report(history, saved)
    else:
        if settings is None:
            raise ValueError('settings are required for a new session')
        session = PanelSession(history, settings)
    events = []
    operations = request.get('operations', [{'action': 'run'}])
    if not isinstance(operations, list):
        raise ValueError('operations must be a JSON array')
    for operation in operations:
        action = operation['action']
        if action == 'advance':
            result = session.advance()
        elif action == 'run':
            session.run()
            result = {'next_date': session.next_date, 'rows': len(session.timeline)}
        elif action == 'report':
            result = {'next_date': session.next_date, 'rows': len(session.timeline)}
        elif action == 'attach_evidence':
            evidence = operation['request']
            if isinstance(evidence, str):
                evidence = json.loads(path(evidence).read_text(encoding='utf-8'))
            result = session.attach_evidence(evidence, applicability=operation['applicability'])
        elif action == 'evidence_step':
            result = session.evidence_step(operation['workflow_action'], operation.get('node'))
        elif action == 'decide_repair':
            if type(operation['accept']) is not bool:
                raise ValueError('accept must be a JSON boolean')
            result = session.decide_repair(operation['accept'],
                method=operation.get('method', 'rolling_base_GQ'), as_of=operation.get('as_of'))
        elif action == 'export':
            result = {'directory': str(session.export(path(operation['directory']),
                         compress=operation.get('compress', False)))}
        else:
            raise ValueError(f'unknown action: {action}')
        events.append({'action': action, 'result': result})
    return clean({'status': 'ok', 'report': session.report(), 'events': events})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-directory', type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.input.read_text(encoding='utf-8'))
        result = execute(request, directory=args.base_directory or args.input.resolve().parent)
    except Exception as exc:
        args.output.write_text(json.dumps({'status': 'error', 'error_type': type(exc).__name__,
                                          'message': str(exc)}, allow_nan=False)+'\n', encoding='utf-8')
        raise SystemExit(1)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
