"""Experimental finite-catalogue joint reporter; does not simulate source laws."""
import math
PAIR_CATALOGUE=('N','CL_Q025','CL_Q050','CL','CL_Q100')
LATENT_CATALOGUE=('N','CM_C025','CM_C050','CM_C075','CM_C100')
ALPHA=.025; TARGET=79; CAP=128


def _num(x,name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(float(x)):
        raise ValueError(f'{name} must be finite numeric')
    return float(x)


def component_bank(attempts, component):
    """Derive the first79 accepted IDs for one component from shared attempts."""
    if component not in ('pair','latent'):raise ValueError('component must be pair or latent')
    if not isinstance(attempts,list) or len(attempts)>CAP:raise ValueError('proposal attempts exceed fixed 128 cap')
    accepted=[];ids=[]
    for i,row in enumerate(attempts):
        if row.get('replicate_id')!=i:raise ValueError('shared proposal IDs must be sequential and identical across components')
        if not isinstance(row,dict) or not isinstance(row.get('pair'),dict) or not isinstance(row.get('latent'),dict):
            raise ValueError('each proposal must carry both component receipts')
        pair_q=row['pair'].get('qualified');latent_q=row['latent'].get('qualified')
        if type(pair_q) is not bool or type(latent_q) is not bool:
            raise ValueError('proposal qualification flags must be explicit booleans; technical errors are not U')
        if latent_q and not pair_q:raise ValueError('FULL-w qualification requires the original paired endpoint')
        q=row[component]['qualified']
        score=row[component].get('score')
        if q is True:
            value=_num(score,component+' score')
            if component=='latent' and value<0:raise ValueError('FULL-w SSE cannot be negative')
            if len(accepted)<TARGET:
                accepted.append(value);ids.append(i)
        elif q is False:
            if score is not None:raise ValueError('ineligible proposal cannot enter a component score bank')
        else:raise ValueError('each component attempt needs an explicit qualification boolean')
    return dict(status='complete' if len(accepted)==TARGET else 'underfilled',proposals=len(attempts),
                qualified=len(accepted),accepted_ids=ids,values=accepted)


def rank_p(observed,values,tail):
    if tail not in ('two-sided','upper'):raise ValueError('tail must be one of the two declared ranks')
    obs=_num(observed,'observed');vals=[_num(x,'bank score') for x in values]
    if len(vals)!=TARGET:raise ValueError('rank requires exactly79 component-qualified proposals')
    lo=(1+sum(x<=obs for x in vals))/80;hi=(1+sum(x>=obs for x in vals))/80
    return min(1.,2*min(lo,hi)) if tail=='two-sided' else hi


def component_result(catalogue,banks,observed,component,tail):
    rows=[]
    for name in catalogue:
        b=banks.get(name)
        if observed is None or b is None or b.get('status')!='complete' or b.get('qualified')!=79:
            rows.append(dict(candidate=name,status='unresolved',p=None));continue
        p=rank_p(observed,b['values'],tail)
        rows.append(dict(candidate=name,status='excluded' if p<=ALPHA else 'retained',p=p))
    positive=all(r['status']=='excluded' for r in rows)
    return dict(status='component_involvement' if positive else 'unknown',alpha=ALPHA,tail=tail,
                statistic='paired FULL-x signed contrast' if component=='pair' else 'FULL-w residual SSE',
                candidate_results=rows)


def build_report(observed,attempts_by_candidate):
    if not isinstance(observed,dict) or not isinstance(attempts_by_candidate,dict):raise ValueError('observed and candidate attempts must be mappings')
    allowed=set(PAIR_CATALOGUE+LATENT_CATALOGUE)
    if set(attempts_by_candidate)-allowed:raise ValueError('undeclared source-catalogue candidate')
    pair=observed.get('pair',{});latent=observed.get('latent',{})
    if type(pair.get('qualified')) is not bool or type(latent.get('qualified')) is not bool:
        raise ValueError('observed qualifications must be explicit booleans; technical errors are not U')
    if latent.get('qualified') is True and pair.get('qualified') is not True:
        raise ValueError('observed FULL-w qualification requires pair qualification')
    if pair.get('qualified') is False and pair.get('score') is not None:raise ValueError('pair U cannot release score')
    if latent.get('qualified') is False and latent.get('score') is not None:raise ValueError('latent U cannot release score')
    if latent.get('qualified') is True and _num(latent.get('score'),'observed FULL-w SSE')<0:raise ValueError('observed FULL-w SSE cannot be negative')
    pairbanks={};latentbanks={}
    for candidate,attempts in attempts_by_candidate.items():
        pairbanks[candidate]=component_bank(attempts,'pair')
        latentbanks[candidate]=component_bank(attempts,'latent')
    pair_obs=_num(pair.get('score'),'observed paired contrast') if pair.get('qualified') is True else None
    latent_obs=latent.get('score') if latent.get('qualified') is True else None
    return dict(model_involvement=component_result(PAIR_CATALOGUE,pairbanks,pair_obs,'pair','two-sided'),
                latent_involvement=component_result(LATENT_CATALOGUE,latentbanks,latent_obs,'latent','upper'),
                catalogues=dict(no_model=PAIR_CATALOGUE,no_latent=LATENT_CATALOGUE),alpha_each=ALPHA,
                shared_proposal_cap=CAP,per_component_target=TARGET,
                scope='Componentwise finite-catalogue ranks with component-specific qualification; unknown is not absence. No pure four-class label or repeated-time guarantee.')


def joint_component_report(*, alarm_status, source_route=None, observed=None,
                           attempts_by_candidate=None):
    """Report both components after an explicit first-stage alarm.

    This is the fixed 79/128, .025-per-component finite-catalogue procedure.
    Caller supplies matched conditional proposals, not arbitrary bootstrap
    draws. A route must have been chosen from released pre-source history;
    this arithmetic interface cannot verify that chronology or kernel truth.
    """
    allowed=('triggered','no_alarm','support_unavailable','technical_invalid','not_run')
    if alarm_status not in allowed:
        raise ValueError('unknown first-stage status')
    result=dict(scope='experimental_joint_finite_catalogue',alarm_status=alarm_status,
        model_involvement='unknown',latent_involvement='unknown',details=None,
        unresolved_reason=None,source_probability=None,posterior_probability=None,
        conditional_assumptions_verified=False,eta_status='not_supplied',
        source_route=source_route,
        assumptions='Correct conditional candidate kernels, true-null catalogue coverage, matched component qualification and history-only routing are required.',
        limitation='No pure-source classification, repeated-time control or empirical power guarantee.')
    if alarm_status!='triggered':
        if source_route is not None or observed is not None or attempts_by_candidate:
            raise ValueError('source data cannot be supplied without a first-stage alarm')
        result['unresolved_reason']=alarm_status
        return result
    routes={'no_source_bank':set(), 'paired_component_only':set(PAIR_CATALOGUE),
            'both_components':set(PAIR_CATALOGUE+LATENT_CATALOGUE)}
    if source_route not in routes:raise ValueError('explicit pre-source route required')
    if not isinstance(attempts_by_candidate,dict) or set(attempts_by_candidate)!=routes[source_route]:
        raise ValueError('candidate inputs must match the declared route, retaining underfilled banks')
    details=build_report(observed,attempts_by_candidate)
    if source_route=='no_source_bank' and observed['pair']['qualified']:
        raise ValueError('qualified observed pair contradicts no-source route')
    if source_route=='paired_component_only' and observed['latent']['qualified']:
        raise ValueError('qualified all32 observation contradicts pair-only route')
    result.update(model_involvement=details['model_involvement']['status'],
                  latent_involvement=details['latent_involvement']['status'],details=details)
    if not observed['pair']['qualified']:result['unresolved_reason']='paired_support_unavailable'
    elif not observed['latent']['qualified']:result['unresolved_reason']='latent_support_unavailable'
    elif 'unknown' in (result['model_involvement'],result['latent_involvement']):
        result['unresolved_reason']='candidate_not_excluded_or_bank_unavailable'
    return result
