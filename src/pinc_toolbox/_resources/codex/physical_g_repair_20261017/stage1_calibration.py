"""Corrected stage-1 calibration dependency; historical banks are never loaded.

Uses the original 59 capped scalar/serial slots and first119-of256 paired bank.
The scientific scores, qualification, seeds, budgets and rank rules are unchanged.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CALIBRATION_ROOT=HERE/'runs/recovery_v003'
TRACK='experiments/ecma_four_tracks_20260924/track1_production'
DEPENDENCIES=(
    'software/pinc_toolbox/src/pinc_toolbox/native_gq.py',
    'software/pinc_toolbox/src/pinc_toolbox/g_estimators.py',
    'software/pinc_toolbox/src/pinc_toolbox/rnd2_counterfactual.py',
    'experiments/GQSimple/paper_rerun/rnd2/simple_adapter.py',
    TRACK+'/persistent_panel_frontier_attempt001/runner.py',
    TRACK+'/panel_four_branch_wrapper_attempt001/panel_wrapper.py',
    TRACK+'/model_union_signed_confirmation3_run01/science_adapter.py',
    TRACK+'/direct_bellman_h8_cal_attempt001/configuration.py',
    TRACK+'/direct_bellman_h8_cal_attempt001/driver.py',
    TRACK+'/direct_bellman_h8_cal_attempt001/monitor_adapter.py',
    TRACK+'/paired_monitor_calibration_attempt001/runner.py',
    TRACK+'/paired_monitor_acquisition_attempt001/paired_monitor.py',
    'codex/physical_g_repair_20261017/stage1_calibration.py',
)


def set_root(path):
    global CALIBRATION_ROOT
    CALIBRATION_ROOT=Path(path).resolve()


def dependency_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in DEPENDENCIES}


def _receipt(group):
    folder=CALIBRATION_ROOT/group/'unit_0000'
    if not (folder/'STATE.json').is_file():
        raise RuntimeError('Corrected stage-1 calibration required first: '+str(folder))
    state=json.loads((folder/'STATE.json').read_text())
    if state.get('execution_status')!='terminal':
        raise RuntimeError('Corrected calibration is not terminal: '+str(folder))
    recorded=json.loads((folder/'DEPENDENCY_HASHES.json').read_text())
    if recorded!=dependency_hashes():
        raise RuntimeError('Corrected calibration code dependencies changed; do not reuse its bank')
    path=folder/'artifacts/RESULT.json'
    return json.loads(path.read_text()),dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_thresholds(*,require_paired=True):
    result,provenance=_receipt('calibration_scalar_serial')
    if result.get('status')!='complete' or not result.get('R_execution_complete'):
        raise RuntimeError('Corrected scalar/serial bank unavailable; no historical fallback')
    slots=result.get('slots',[])
    if len(slots)!=59 or any(s.get('status')!='accepted' for s in slots):
        raise RuntimeError('Need all original59 accepted scalar/serial slots')
    assigned=result.get('assigned',[])
    expected=[(j,k+1,3252601000+3*j+k) for j in range(59) for k in range(3)]
    if [(r.get('slot'),r.get('attempt'),r.get('seed')) for r in assigned]!=expected:
        raise RuntimeError('Original capped scalar/serial seed assignment changed')
    scalar=[];serial=[]
    for s in slots:
        matches=[r for r in assigned if r['slot']==s['slot'] and r['attempt']==s['attempt']]
        if len(matches)!=1 or matches[0]['seed']!=s['seed']:
            raise RuntimeError('Accepted slot has no exact assigned seed/attempt')
        accepted=(matches[0].get('result') or {})
        if accepted.get('status')!='available' or accepted.get('completed_dates')!=8:
            raise RuntimeError('Accepted calibration monitor is incomplete')
        earlier=[r for r in assigned if r['slot']==s['slot'] and r['attempt']<s['attempt']]
        if any((r.get('result') or {}).get('status')=='available' for r in earlier):
            raise RuntimeError('First-available slot selection changed')
        scalar.append(accepted['scores']['scalar']);serial.append(accepted['scores']['serial'])
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0 for x in scalar+serial):
        raise RuntimeError('Nonfinite/negative complete stage1 score')
    if max(serial)!=result.get('threshold'):
        raise RuntimeError('Serial strict-maximum readback mismatch')
    thresholds=dict(serial=max(serial),scalar=max(scalar),corrected_calibration_provenance={'scalar_serial':provenance})
    if require_paired:
        paired,pp=_receipt('calibration_paired')
        bank=paired.get('bank',[]);attempts=paired.get('attempts',[])
        if paired.get('status')!='bank_complete' or paired.get('qualified')!=119 or len(bank)!=119:
            raise RuntimeError('Corrected paired bank unavailable; require119 qualified within256')
        if not 119<=len(attempts)<=256 or [(a.get('ordinal'),a.get('seed')) for a in attempts]!=list(enumerate(range(3252760000,3252760000+len(attempts)))):
            raise RuntimeError('Original paired bank assignment changed')
        scores=[a['score'] for a in attempts if a.get('qualification')=='complete_8_date_pair']
        if scores!=bank or len(scores)!=119 or attempts[-1].get('qualification')!='complete_8_date_pair':
            raise RuntimeError('First119 paired qualifications/readback mismatch')
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in bank):
            raise RuntimeError('Nonfinite paired calibration score')
        thresholds['paired_bank']=bank
        thresholds['corrected_calibration_provenance']['paired']=pp
    return thresholds


def patch_worker_source(source):
    """Replace only the stale bank-input seam in the existing immutable driver."""
    start=source.index('    xread=json.loads(')
    stop=source.index('    started=time.monotonic()',start)
    old=source[start:stop]
    if old.count('h8_xscalar_readback_attempt001/READBACK.json')!=1 or old.count('paired_monitor_calibration_attempt001/execution001/RESULT.json')!=1:
        raise RuntimeError('Original stage1 input seam changed; inspect before execution')
    return source[:start]+'    thresholds=load_corrected_stage1()\n'+source[stop:]
