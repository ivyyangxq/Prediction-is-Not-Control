"""Original source-window energies and four exact signed ranks; no simulation."""
import math
import numpy as np
from pinc_toolbox.residual_window_energy import window_slope_energy
from pinc_toolbox.stopped_candidate_rank import stopped_candidate_rank

def branch_score(branch):
    result=dict(full_window_complete=False,score_valid_for_rank=False,scalar=None,serial=None,raw_status=None if branch is None else branch['status'])
    if branch is None:return result
    traces=branch['trace'];dates=[];ex=[];ew=[]
    for tr in traces:
        if not tr.get('released'):continue
        t=tr['time'];issued=tr['issued'];r=issued['main'];nxt=tr['next_state']['physical']
        if not (tr['release_time']==t+1 and issued['time']==issued['fit_as_of']==r['time']==r['fit_as_of']==r['base_fit_as_of']==r['full_fit_as_of']==t):raise ValueError('source receipt clock mismatch')
        if tr['action']!=r['action'] or r['full_state_prediction']!=r['full_physical_prediction'][0]:raise ValueError('source receipt action/projection mismatch')
        values=[nxt[0],nxt[1],r['full_state_prediction'],r['full_physical_prediction'][1]]
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite source receipt')
        dates.append(t);ex.append(nxt[0]-r['full_state_prediction']);ew.append(nxt[1]-r['full_physical_prediction'][1])
    if dates!=list(range(40,40+len(dates))) or len(dates)>8:raise ValueError('invalid source release sequence')
    if branch['status']!='complete':return result
    if dates!=list(range(40,48)) or branch['completed_steps']!=8:raise ValueError('complete status without eight releases')
    def energy(e,w):return float(window_slope_energy(np.array(e)[None,:,None],np.array(w)[None,:],residual_left_inverse=np.eye(1),windows=(8,),eligible=np.ones((1,8,1),bool))['scores'][0,0])
    scores=dict(scalar=energy(ex,np.ones(8)),serial=energy(ew,[0.]+ew[:-1]))
    if not all(math.isfinite(v) for v in scores.values()):raise ValueError('nonfinite source energy')
    return dict(result,full_window_complete=True,score_valid_for_rank=True,**scores)

def candidate_decision(observed,bank):
    if len(bank)!=79:raise ValueError('retain all 79 assigned slots')
    for row in [observed]+bank:
        complete=row['full_window_complete'];valid=row['score_valid_for_rank']
        if type(complete) is not bool or type(valid) is not bool or complete!=valid:raise ValueError('protocol requires matching completion and validity flags')
        if not complete and (row['scalar'] is not None or row['serial'] is not None):raise ValueError('non-C scores must be null')
        if complete and not all(isinstance(row[k],(float,int)) and not isinstance(row[k],bool) and math.isfinite(row[k]) for k in ('scalar','serial')):raise ValueError('C requires two finite scores')
    ranks={}
    for stat in ('scalar','serial'):
        for sign in (1,-1):
            signed=lambda row:None if row[stat] is None else sign*row[stat]
            ranks[stat+('_upper' if sign==1 else '_lower')]=stopped_candidate_rank(observed_score=signed(observed),observed_full_window_complete=observed['full_window_complete'],observed_score_valid_for_rank=observed['score_valid_for_rank'],calibration_scores=[signed(r) for r in bank],calibration_full_window_complete=[r['full_window_complete'] for r in bank],calibration_score_valid_for_rank=[r['score_valid_for_rank'] for r in bank],alpha=.0125)
    return dict(excluded=any(r['rejected'] for r in ranks.values()),unresolved=not observed['full_window_complete'],ranks=ranks)
