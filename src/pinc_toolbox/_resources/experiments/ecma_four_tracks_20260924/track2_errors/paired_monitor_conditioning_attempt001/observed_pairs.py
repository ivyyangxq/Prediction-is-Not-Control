"""Assemble all observed probe information for candidate conditioning.
Inputs are sanitized public records, not environment objects or truth labels.
"""
import math

def assemble_pairs(main_transitions, probe_observations):
    main={}
    for row in main_transitions:
        key=(row['path'],row['time'])
        if key in main:raise ValueError('duplicate MAIN transition')
        main[key]=row
    grouped={k:[] for k in main};refusals=[]
    for row in probe_observations:
        key=(row['path'],row['time'])
        if key not in main:raise ValueError('probe without corresponding MAIN transition')
        if row['status']=='support_refusal':
            refusals.append(dict(row));continue
        if row['status']!='complete':raise ValueError('non-support probe error cannot be ignored')
        grouped[key].append(row)
    pairs=[]
    for key,m in main.items():
        rows=[m]+grouped[key]
        outcomes={}
        for row in rows:
            if row['release_time']!=row['time']+1:raise ValueError('invalid release chronology')
            if not math.isclose(row['state_x'],m['state_x'],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('initial states differ')
            a=row['action'];x=row['next_x']
            if not all(math.isfinite(v) for v in (a,x,row['state_x'])):raise ValueError('nonfinite observation')
            if not 0<=a<=.1 or abs(round(a*100)-a*100)>1e-10:raise ValueError('action outside grid')
            if a in outcomes and not math.isclose(outcomes[a],x,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('same action produced incompatible same-noise observations')
            outcomes[a]=x
        if len(outcomes)<2:continue
        low,high=min(outcomes),max(outcomes)
        correction=lambda a:.8*a-2*a*a+.5*m['state_x']*a
        slope=((outcomes[high]-correction(high))-(outcomes[low]-correction(low)))/(high-low)
        for a,x in outcomes.items():
            fitted=outcomes[low]-correction(low)+slope*(a-low)+correction(a)
            if not math.isclose(x,fitted,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('observed endpoints violate common-noise affine action relation')
        pairs.append(dict(path=key[0],time=key[1],release_time=m['release_time'],state_x=m['state_x'],
            low_action=low,high_action=high,low_next_x=outcomes[low],high_next_x=outcomes[high]))
    return pairs,refusals
