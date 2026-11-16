"""Original scalar/serial formulas for fixed eight-date reset-probe streams."""
import math
import numpy as np
from pinc_toolbox.residual_window_energy import window_slope_energy

def score(rows,start=40):
 out=dict(complete=False,scalar=None,serial=None,signed_serial_numerator=None,serial_denominator=None)
 if len(rows)!=8 or [r['time'] for r in rows]!=list(range(start,start+8)) or any(r['status']!='complete' for r in rows):return out
 ex=np.array([r['full_x_error'] for r in rows],float);ew=np.array([r['full_w_error'] for r in rows],float)
 if not np.isfinite(ex).all() or not np.isfinite(ew).all():raise ValueError('nonfinite original residual')
 lag=np.r_[0.,ew[:-1]]
 def energy(e,w):return float(window_slope_energy(e[None,:,None],w[None],residual_left_inverse=np.eye(1),windows=(8,),eligible=np.ones((1,8,1),bool))['scores'][0,0])
 return dict(complete=True,scalar=energy(ex,np.ones(8)),serial=energy(ew,lag),signed_serial_numerator=float(np.dot(ew,lag)),serial_denominator=float(np.dot(lag,lag)))

def pair(low,high):
 out=dict(time=low['time'],low_action=.02,high_action=.08,status='unavailable',actual_x_difference=None,predicted_x_difference=None,x_residual_difference=None,x_identity_error=None,actual_w_difference=None,predicted_w_difference=None,w_residual_difference=None,w_identity_error=None,instant_reward_difference=None)
 if low['time']!=high['time']:raise ValueError('pair dates differ')
 if low['status']!='complete' or high['status']!='complete':return out
 def main_issued(r):return r['issued']['main']
 for j,key in ((0,'x'),(1,'w')):
  actual=high['next_physical'][j]-low['next_physical'][j]
  predicted=main_issued(high)['full_physical_prediction'][j]-main_issued(low)['full_physical_prediction'][j]
  error=high['full_'+key+'_error']-low['full_'+key+'_error'];difference=error-(actual-predicted)
  if not all(math.isfinite(v) for v in (actual,predicted,error,difference)) or abs(difference)>1e-12:raise ValueError('paired residual identity failure')
  out.update({f'actual_{key}_difference':actual,f'predicted_{key}_difference':predicted,f'{key}_residual_difference':error,f'{key}_identity_error':difference})
 if out['actual_w_difference']!=0:raise ValueError('same-date probes changed actual proxy')
 out.update(status='available',instant_reward_difference=high['reward']-low['reward'])
 return out
