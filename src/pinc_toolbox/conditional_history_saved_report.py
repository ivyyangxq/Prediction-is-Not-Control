"""Pure reporting of a supplied conditional-history protocol terminal object.

Standard library only. No experiment imports, file reads, fitting or sampling.
Validation checks internal consistency, not the truth of maintained assumptions.
"""
from copy import deepcopy
from fractions import Fraction
import json
import math

CELLS=('N','CL','CM','MIX')
TAILS={'N':('upper','upper'),'CL':('upper','lower'),
       'CM':('lower','upper'),'MIX':('lower','lower')}
U={'initialization_support_refusal','initialization_identification_unavailable',
   'main_support_refusal','main_identification_unavailable'}
ASSUMPTIONS=('finite_fully_specified_catalogue','complete_public_history_fixed',
             'fixed_detector_alarm_measurable','correct_candidate_conditional_kernels',
             'independent_CAL_and_observed_future','original_scores_tails_and_qualification',
             'predeclared_capped_first_available_sampling')

class InvalidReportInput(ValueError):
    pass

def require(condition,message):
    if not condition:raise InvalidReportInput(message)

def mapping(value,name):
    require(isinstance(value,dict),name+' must be an object')
    return value

def number(value,name):
    require(type(value) in (int,float) and math.isfinite(value),name+' must be finite real, not bool')
    return value

def boolean(value,name):
    require(type(value) is bool,name+' must be boolean')
    return value

def integer(value,name,low,high):
    require(type(value) is int and low<=value<=high,name+' is out of range')
    return value

def score(value,name):
    require(number(value,name)>=0,name+' must be nonnegative')
    return value

def bits(retained):
    if not retained:return 'unresolved','unresolved'
    values=set(retained)
    model='present' if values<={'CM','MIX'} else 'absent' if values<={'N','CL'} else 'unresolved'
    latent='present' if values<={'CL','MIX'} else 'absent' if values<={'N','CM'} else 'unresolved'
    return model,latent

def mixed_status(retained):
    if retained==['MIX']:return 'present'
    if retained and 'MIX' not in retained:return 'absent'
    return 'unresolved'

def monitor_record(terminal):
    rows=terminal.get('monitors')
    require(isinstance(rows,list) and len(rows) in (1,128),'require one look or 128 preassigned monitor records')
    selected=terminal.get('selected');alarm_indices=[];available=unavailable=0
    for i,row in enumerate(rows):
        mapping(row,'monitor');status=row.get('status');h=row.get('H')
        if status=='available':
            require(type(h) is bool and row.get('completed_dates')==8,'available monitor needs Boolean H and eight dates')
            score(mapping(row.get('scores'),'monitor scores').get('serial'),'monitor serial')
            available+=1
            if h:alarm_indices.append(i)
        elif status in U:
            require(h is None,'unavailable monitor must retain H=null');unavailable+=1
        elif status=='not_run_first_alarm_selected':
            require(h is None,'unrun monitor cannot have H')
        else:raise InvalidReportInput('invalid terminal monitor status: '+str(status))
    if selected is None:
        require(not alarm_indices and all(r['status']=='available' or r['status'] in U for r in rows),
                'no selected alarm contradicts monitor ledger')
        require(terminal.get('H') is None,'no_alarm_within_cap aggregate H must remain null')
        status='no_alarm' if available else 'unavailable'
    else:
        integer(selected,'selected',0,len(rows)-1)
        require(alarm_indices==[selected] and terminal.get('H') is True,'selected alarm must be unique and first')
        require(all(r['status']=='available' or r['status'] in U for r in rows[:selected]),'invalid pre-alarm ledger')
        require(all(r['status']=='not_run_first_alarm_selected' for r in rows[selected+1:]),'monitor continued after first alarm')
        status='alarm'
    return dict(status=status,H=terminal.get('H') if len(rows)>1 else rows[0]['H'],
                scope='single_look' if len(rows)==1 else 'first_alarm_within_128_histories',
                release_time=40,selected=selected,planned_histories=len(rows),
                available_looks=available,unavailable_looks=unavailable,
                statistical_coverage='not_verified_by_reporting_adapter',
                familywise_scan_guarantee='not_established')

def validate_banks(banks):
    mapping(banks,'banks');require(set(banks)==set(CELLS),'banks require exactly N,CL,CM,MIX')
    for cell,bank in banks.items():
        mapping(bank,'bank');require(bank.get('cell')==cell,'bank candidate identity mismatch')
        require(bank.get('fatal') is False,'fatal bank cannot supply successful inference')
        require(bank.get('target')==39 and bank.get('cap')==128,'bank target/cap mismatch')
        proposals=integer(bank.get('proposals'),'proposals',0,128)
        for name in ('accepted','available','accepted_rows'):
            require(set(mapping(bank.get(name),name))=={'A','B'},'bank branches must be A and B')
        for branch in ('A','B'):
            rows=bank['accepted_rows'][branch]
            require(isinstance(rows,list),'accepted rows must be an array')
            n=integer(bank['accepted'][branch],'accepted count',0,39)
            require(n==len(rows) and n<=proposals,'accepted count disagrees with rows/proposals')
            require(boolean(bank['available'][branch],'bank available')==(n==39),'bank availability mismatch')
            positions=[];identities=[]
            for row in rows:
                mapping(row,'accepted row');score(row.get('score'),'CAL score')
                positions.append(integer(row.get('proposal_position'),'proposal_position',1,proposals))
                identity=row.get('seed');require(type(identity) in (str,int),'proposal identity required')
                identities.append(identity)
            require(positions==sorted(set(positions)) and len(set(identities))==len(identities),
                    'accepted rows must retain unique ordered proposal identities')
        full=all(bank['available'].values())
        require(boolean(bank.get('both_targets_reached'),'both_targets_reached')==full,'target completion mismatch')
        require(boolean(bank.get('cap_incomplete'),'cap_incomplete')==(proposals==128 and not full),'cap status mismatch')
        require(full or proposals==128,'terminal bank stopped before both targets or cap')

def validate_inference(terminal):
    banks=terminal.get('banks');validate_banks(banks)
    observed=mapping(terminal.get('observed'),'observed')
    require(observed.get('H') is True,'selected continuation must retain fixed H=true')
    require(observed.get('status') in ('complete','MAIN_unavailable'),'unknown/fatal observed status')
    require(observed.get('failure') is None,'compact observed must not contain an unresolved failure')
    ca=boolean(observed.get('C_A'),'C_A');cb=boolean(observed.get('C_B'),'C_B')
    require(not ca or cb,'C_A must imply C_B')
    main=integer(observed.get('main_source_completed'),'MAIN dates',0,8)
    require(observed.get('monitor_completed')==8,'observed monitor must be complete')
    require(not (ca or cb) or main==8 and observed['status']=='complete','qualified branch lacks complete MAIN')
    require((observed['status']=='complete')==(main==8),'observed status disagrees with MAIN completion')
    for branch in ('A','B'):
        if observed['C_'+branch]:score(observed.get(branch),'observed '+branch)
        else:require(observed.get(branch) is None,'unavailable observed branch must have null score')
    inference=mapping(terminal.get('inference'),'inference')
    require(inference.get('H') is True,'inference H inconsistent with selected history')
    candidates=inference.get('candidates')
    require(isinstance(candidates,list) and len(candidates)==4,'inference needs four unique candidates')
    ids=[mapping(c,'candidate').get('candidate') for c in candidates]
    require(len(set(ids))==4 and set(ids)==set(CELLS),'duplicate/missing/unknown candidate')
    checked={};unperformed=[]
    for item in candidates:
        cell=item['candidate'];branches=mapping(item.get('branches'),'candidate branches')
        require(set(branches)=={'A','B'},'candidate needs A and B')
        for branch in ('A','B'):
            p=mapping(branches[branch],'branch result')
            performed=boolean(p.get('test_performed'),'test_performed')
            reasons=[]
            if not observed['C_'+branch]:reasons.append('observed_branch_unavailable')
            if not banks[cell]['available'][branch]:reasons.append('bank_incomplete')
            require(p.get('reasons')==reasons and p.get('reason')==(reasons[0] if reasons else None),
                    'branch unavailable reasons inconsistent with observed/bank')
            require(performed==(not reasons),'test_performed contradicts branch availability')
            if reasons:
                require(type(p.get('p')) in (int,float) and p['p']==1 and p.get('p_exact')=='1',
                        'scientific unavailable must be operational p=1, untested')
                require('tail' not in p,'untested p must not claim a tested tail')
                unperformed.append(dict(candidate=cell,branch=branch,reasons=reasons))
            else:
                tail=TAILS[cell][branch=='B']
                require(p.get('tail')==tail,'candidate tail mismatch')
                values=[r['score'] for r in banks[cell]['accepted_rows'][branch]];value=observed[branch]
                numerator=1+sum(x>=value if tail=='upper' else x<=value for x in values)
                require(p.get('p_exact')==str(numerator)+'/40','p_exact disagrees with original inclusive rank')
                require(number(p.get('p'),'p')==float(Fraction(numerator,40)),'numeric p disagrees with exact rank')
        excluded=any(v['test_performed'] and v['p']<=1/40 for v in branches.values())
        require(boolean(item.get('excluded'),'excluded')==excluded,'candidate exclusion disagrees with branches')
        checked[cell]=deepcopy(item)
    retained=[c for c in CELLS if not checked[c]['excluded']]
    require(inference.get('retained')==retained,'retained set disagrees with exclusions/order')
    model,latent=bits(retained)
    require(inference.get('model')==model and inference.get('latent')==latent,'component bits disagree with retained set')
    expected='branch_unavailable' if any(not any(p['test_performed'] for p in checked[c]['branches'].values()) for c in CELLS) else 'complete'
    require(inference.get('status')==expected,'inference status inconsistent with performed branches')
    return checked,retained,model,latent,unperformed

def adapt(terminal, *, context=None, maintained_assumptions=None):
    """Validate and render supplied protocol output, without reading experiments.

    Context accepts only public references; terminal spec/truth/monitor candidate
    labels are never propagated. Assumptions are caller declarations, not checks.
    """
    mapping(terminal,'terminal')
    context={} if context is None else deepcopy(mapping(context,'context'))
    require(set(context)<={'history_reference','detector_reference'},'context may contain only public history/detector references')
    require(all(isinstance(v,str) and v for v in context.values()),'public references must be nonempty strings')
    declared={} if maintained_assumptions is None else deepcopy(mapping(maintained_assumptions,'assumptions'))
    require(set(declared)<=set(ASSUMPTIONS),'unknown maintained assumption')
    require(all(type(v) is bool for v in declared.values()),'assumptions must be explicit Boolean declarations')
    complete_assumptions=all(declared.get(k) is True for k in ASSUMPTIONS)
    context.update(route='conditional_history_saved_report',catalogue=list(CELLS),
                   scope='saved-terminal reporting only; no conditional sampler')
    def errors(name):
        condition='unconditional; horizon specified by stage1.scope' if name=='detection' else 'conditional_on_alarm; source assertions only'
        targets={
            'detection':('alarm under the declared detection null', 'no alarm under a specified detection alternative by the declared horizon'),
            'model':('assert model present when the model component is absent', 'fail to assert model present when the model component is present; includes unresolved or withheld reports'),
            'latent':('assert latent present when the latent component is absent', 'fail to assert latent present when the latent component is present; includes unresolved or withheld reports'),
            'mixed':('assert mixed_status present when the true law is not MIX', 'fail to assert mixed_status present when the true law is MIX; includes unresolved or withheld reports'),
            'joint':('any false presence assertion for model, latent or mixed; excludes false absence assertions', 'fail to assert a truly present component; includes unresolved or withheld reports')}
        return {kind:dict(target=targets[name][i],conditioning=condition,target_budget=None,proved_bound=None,empirical=None,
                          status='not_established' if kind=='type_i' else 'not_estimated')
                for i,kind in enumerate(('type_i','type_ii'))}
    report=dict(context=context,stage1=dict(status='unavailable'),
        stage2=dict(status='not_entered',model='not_entered',latent='not_entered',mixed=False,
                    mixed_status='unresolved',mixed_boolean_meaning='joint_presence_asserted_only; false_is_not_an_absence_assertion',
                    outcome='not_entered',evidence={},retained=None,trace=dict(visited=[],inherited=[],not_run=list(CELLS))),
        errors={name:errors(name) for name in ('detection','model','latent','mixed','joint')},
        reliability=dict(statistical_coverage='not_verified_by_reporting_adapter',numerical_certification='internal_report_consistency_only',
                         source_probability=dict(status='unavailable',value=None),assumptions=declared,
                         assumptions_verified=False,proof_applicable='not_established'),
        completion=dict(status='complete',failures=[]))
    report['errors']['joint']['resolved_assertion_error']=dict(
        target='any wrong resolved presence or absence assertion for model, latent or mixed',
        conditioning='conditional_on_alarm; source calibration and observed future averaged',
        target_budget=None,proved_bound=None,empirical=None,status='not_established',
        scope='One shared bound; false absence is covered, but unresolved missed detections are not estimated by this bound.')
    status=terminal.get('status')
    require(status in ('complete','no_alarm_within_cap','invalid'),'only protocol terminal states may be reported')
    if status=='invalid':
        require(terminal.get('inference') is None,'invalid campaign cannot carry successful inference')
        require(terminal.get('failure') is not None,'invalid terminal needs explicit failure')
        report['stage2'].update(status='unavailable',outcome='campaign_invalid_no_inference',model='unresolved',latent='unresolved')
        report['completion'].update(status='incomplete',failures=[dict(reason='campaign_invalid')])
        # Preserve only the public alarm bit, not truth-bearing raw failure/spec payloads.
        h=terminal.get('H');require(h is None or type(h) is bool,'invalid H must be bool or null')
        report['stage1']=dict(status='alarm' if h is True else 'unavailable',H=h,scope='invalid_campaign_no_guarantee')
    else:
        require(terminal.get('failure') is None,'successful terminal cannot conceal failure')
        report['stage1']=monitor_record(terminal)
        if status=='no_alarm_within_cap':
            require(terminal.get('selected') is None and terminal.get('observed_future_started') is False,
                    'no-alarm terminal cannot start observed future')
            require(all(terminal.get(k) is None for k in ('observed','inference','banks')),'no-alarm terminal cannot contain source results')
            report['stage2']['outcome']='no_alarm_within_cap'
            if report['stage1']['unavailable_looks']:
                report['completion'].update(status='incomplete',failures=[dict(reason='some_monitors_unavailable')])
        else:
            require(terminal.get('selected') is not None and terminal.get('observed_future_started') is True,
                    'complete source report requires selected alarm and observed future')
            require(not any(r.get('status') in ('fatal','running','not_run_after_fatal') for r in terminal.get('CAL',[])),
                    'complete campaign contains fatal/unfinished CAL rows')
            checked,retained,model,latent,unperformed=validate_inference(terminal)
            report['stage2'].update(status='reported',outcome='empty_retained_set_unresolved' if not retained else terminal['inference']['status'],
                evidence=checked,retained=retained,model=model,latent=latent,mixed=model==latent=='present',
                mixed_status=mixed_status(retained),
                trace=dict(visited=list(CELLS),inherited=[],not_run=[]))
            if unperformed:report['completion'].update(status='incomplete',failures=unperformed)
            if complete_assumptions:
                report['reliability']['proof_applicable']='only_under_explicitly_maintained_assumptions; not_software_verified'
                report['errors']['joint']['resolved_assertion_error'].update(
                    target_budget=.05,proved_bound=.05,status='conditional_on_maintained_assumptions')
                for name in ('model','latent','mixed','joint'):
                    report['errors'][name]['type_i'].update(target_budget=.05,proved_bound=.05,
                        status='conditional_on_maintained_assumptions')
    if report['reliability']['proof_applicable']=='not_established':
        error_scope='No source error bound is established for this report.'
        if status=='invalid':
            error_scope+=' Campaign invalid: source inference is withheld; no candidate p values, retained set or resolved source assertions are reported.'
    else:
        error_scope='Under the explicitly maintained assumptions only, one shared .05 resolved-assertion bound covers wrong presence and wrong absence; these assumptions are not software verified.'
    report['context']['error_scope']=error_scope+' Type I fields cover false presence only; Type II, including unresolved missed detections, remains unestimated. Do not add component budgets. No scan-familywise or combined detector/source bound.'
    report['context']['execution_scope']='No fixed-bank, successful-execution/publication or resolved-report conditional guarantee; resource failures withhold inference.'
    report['semantic_disclosure']='p values are inclusive ranks, not posterior source probabilities. Operational p=1 with test_performed=false is abstention. Known finite catalogue and correct conditional kernels are maintained assumptions. No power claim.'
    if status=='invalid':
        report['semantic_disclosure']='Source inference is withheld because the campaign is invalid. Source probabilities are unavailable. No power claim.'
    try:json.dumps(report,allow_nan=False)
    except (TypeError,ValueError) as exc:raise InvalidReportInput('report is not finite JSON') from exc
    return report

def render_report(report):
    lines=['Stage 1: '+report['stage1']['status']+' ('+report['stage1'].get('scope','unspecified')+')',
           'Stage 2: '+report['stage2']['outcome'],
           'Retained: '+str(report['stage2']['retained']),
           'Model: '+report['stage2']['model']+'; latent: '+report['stage2']['latent'],
           'Mixed status: '+report['stage2']['mixed_status']+'; joint_presence_asserted='+str(report['stage2']['mixed'])+' (false is not a negative assertion)',
           report['semantic_disclosure'],report['context']['error_scope'],
           'Proof applicability: '+report['reliability']['proof_applicable'],
           'Maintained assumptions verified by software: false']
    for c,item in report['stage2']['evidence'].items():
        for b,p in item['branches'].items():
            lines.append(f"{c}/{b}: p={p['p_exact']}; tested={p['test_performed']}; reasons={p['reasons']}")
    return '\n'.join(lines)+'\n'
