"""Optional support-only routing for caller-declared input-only matched hosts."""
import math

BRANCHES=('main','fixed_action','fixed_public_environment','both_fixed')
PAIRS={'primary':BRANCHES[:2],'secondary':BRANCHES[2:]}

def finite(value):
 return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def route_action_pairs(report,*,applicability,verified_prepare_failures=None):
 """Primary wins; only declared support refusal can enable secondary fallback.

 verified_prepare_failures contains independently verified finite-input
 identification classification. Raw software failures otherwise invalidate the
 entire execution, including failures outside the selected pair.
 """
 if applicability!='input_only_matched_physical_paths':
  raise ValueError('declare input_only_matched_physical_paths applicability explicitly')
 prepare_failures=verified_prepare_failures or {};invalid=[];reasons={};pairs={}
 bs=report['branches']
 start=report.get('start_time');steps=report.get('steps');beta=report.get('beta');end=report.get('final_release_time')
 valid_window=(type(start) is int and type(steps) is int and steps>0 and type(end) is int and end==start+steps and finite(beta) and 0<=beta<=1 and report.get('mode')=='synchronous_main_tape')
 if not valid_window:
  return dict(status='invalid',selected_branch=None,value=None,reason='wrong_declared_window_discount_or_mode',primary=None,secondary=None,invalid_records=[dict(branch=None,reason='wrong_declared_window_discount_or_mode')],original_all_four_status=report.get('comparison',{}).get('status'),applicability=applicability,inference=dict(status='unavailable',reason='no_matched_routed_calibration_supplied'))
 for name in BRANCHES:
  b=bs[name];status=b['status'];fail=b.get('failures',[])
  if status=='complete':
   trace=b.get('trace',[])
   valid=(finite(b.get('discounted_return')) and type(b.get('completed_steps')) is int and b.get('completed_steps')==steps and not fail and len(trace)==steps and all(type(t.get('time')) is int and t.get('time')==start+i and type(t.get('release_time')) is int and t.get('release_time')==start+i+1 and t.get('released') is True and finite(t.get('reward')) for i,t in enumerate(trace)))
   if valid:
    computed=sum(beta**i*t['reward'] for i,t in enumerate(trace))
    valid=finite(computed) and math.isclose(computed,b['discounted_return'],rel_tol=1e-12,abs_tol=1e-12)
   reasons[name]='complete' if valid else 'invalid_complete_record'
   if not valid:invalid.append(dict(branch=name,reason=reasons[name]))
  elif status=='support_refusal':
   valid=bool(fail) and all(f.get('type')=='SupportError' and type(f.get('time')) is int and start<=f['time']<=end and isinstance(f.get('reason'),str) for f in fail)
   reasons[name]='support_refusal' if valid else 'unverified_support_refusal'
   if not valid:invalid.append(dict(branch=name,reason=reasons[name]))
  elif status=='software_failure':
   p=prepare_failures.get(name,{})
   verified=p.get('classification',{}).get('status')=='identification_unavailable' and bool(fail) and all(f.get('time')==p.get('time') and f.get('type')==p.get('error_type') and f.get('reason')==p.get('reason') for f in fail)
   reasons[name]='identification_unavailable' if verified else 'software_failure'
   if not verified:invalid.append(dict(branch=name,reason='software_failure',failures=fail))
  elif status=='main_unavailable':reasons[name]='main_unavailable'
  else:
   reasons[name]='unknown_branch_status';invalid.append(dict(branch=name,reason=reasons[name],status=status))
 for name in BRANCHES:
  if reasons[name]=='main_unavailable':
   cause=reasons['main']
   if name=='main' or cause in ('complete','main_unavailable'):
    invalid.append(dict(branch=name,reason='main_unavailable_without_main_cause'))
   reasons[name]='main_unavailable_due_to_'+cause
 for label,names in PAIRS.items():
  complete=all(reasons[n]=='complete' for n in names)
  value=bs[names[0]]['discounted_return']-bs[names[1]]['discounted_return'] if complete else None
  if complete and not finite(value):invalid.append(dict(branch=label,reason='nonfinite_return_difference'));complete=False;value=None
  unavailable=[n for n in names if reasons[n]!='complete']
  support=bool(unavailable) and all(reasons[n] in ('support_refusal','main_unavailable_due_to_support_refusal') for n in unavailable)
  pairs[label]=dict(status='complete' if complete else 'support_unavailable' if support else 'unavailable',available=complete,value=value,required_branches=list(names),unavailable_branches=unavailable,branch_reasons={n:reasons[n] for n in names},failures={n:bs[n].get('failures',[]) for n in names if reasons[n]!='complete'})
 selected=None
 if invalid:status='invalid';reason='invalid_execution_record'
 elif pairs['primary']['available']:selected='primary';status='available';reason='primary_complete'
 elif pairs['primary']['status']=='support_unavailable' and pairs['secondary']['available']:selected='secondary';status='available';reason='primary_support_refusal_secondary_complete'
 else:status='unavailable';reason='primary_not_support_refusal' if pairs['primary']['status']!='support_unavailable' else 'secondary_incomplete'
 return dict(status=status,selected_branch=selected,value=None if selected is None else pairs[selected]['value'],reason=reason,primary=pairs['primary'],secondary=pairs['secondary'],branch_reasons=reasons,invalid_records=invalid,original_all_four_status=report['comparison']['status'],applicability=applicability,applicability_status='caller_declared_not_verified_by_reporting',selection_event='primary_complete or (primary_support_refusal and secondary_complete), within valid execution and the declared entry event',inference=dict(status='unavailable',reason='no_matched_routed_calibration_supplied'),scope='fixed routing of existing contrasts; no source calibration, p-value, or classification')
