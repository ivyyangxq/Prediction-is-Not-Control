"""Prospective panel availability classifier prototype; no fitting or sampling."""
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parent/'runtime_source'
sys.path.insert(0,str(SOURCE))
import numpy as np
from pinc_toolbox.g import IdentificationError,SupportError

KNOWN_IDENTIFICATION={
 'fewer observations than declared columns':'row_count',
 'state dictionary is rank deficient':'state_rank',
 '[B,d] is rank deficient':'joint_rank',
 'D_perp is rank deficient':'residualized_rank',
}

def classify(exc, *, raw_inputs_finite, B=None, D=None, Y=None):
 """Classify an ALREADY raised production exception; never skip a fit.

 Arrays must be those actually passed to the failed full/base G solver.
 Caller checks released data clocks/shapes before fitting and supplies the
 finite-input flag. Ambiguous exceptions remain fatal even when their text
 resembles one of these causes. No window, tolerance or columns are changed.
 """
 fatal={'status':'fatal','reason':'software_or_resource_failure','exception_type':type(exc).__name__,'message':str(exc)}
 if raw_inputs_finite is not True:return fatal
 if isinstance(exc,SupportError):return {**fatal,'status':'support_unavailable','reason':'native_support_or_domain'}
 if not isinstance(exc,IdentificationError) or str(exc) not in KNOWN_IDENTIFICATION:return fatal
 try:
  b=np.asarray(B,dtype=float);d=np.asarray(D,dtype=float);y=np.asarray(Y,dtype=float)
  if any(x.ndim!=2 or not np.isfinite(x).all() for x in (b,d,y)):return fatal
  n=len(b)
  if n==0 or len(d)!=n or len(y)!=n or min(b.shape[1],d.shape[1],y.shape[1])==0:return fatal
  reason=KNOWN_IDENTIFICATION[str(exc)];evidence={'rows':n,'B_columns':b.shape[1],'D_columns':d.shape[1]}
  if reason=='row_count':confirmed=n<b.shape[1]+d.shape[1]
  else:
   if n<b.shape[1]+d.shape[1]:return fatal
   bs=np.max(np.abs(b),axis=0);ds=np.max(np.abs(d),axis=0)
   if not np.isfinite(bs).all() or not np.isfinite(ds).all() or np.any(bs<=0) or np.any(ds<=0):return fatal
   bn=b/bs;dn=d/ds
   projection=np.linalg.lstsq(bn,np.column_stack((dn,y)),rcond=None);brank=int(projection[2]);evidence['B_rank']=brank
   if reason=='state_rank':confirmed=brank<b.shape[1]
   else:
    if brank<b.shape[1]:return fatal
    joint=np.c_[bn,dn];jrank=int(np.linalg.lstsq(joint,np.zeros((n,1)),rcond=None)[2]);evidence['joint_rank']=jrank
    if reason=='joint_rank':confirmed=jrank<joint.shape[1]
    else:
     if jrank<joint.shape[1]:return fatal
     dp=dn-bn@projection[0][:,:d.shape[1]]
     if not np.isfinite(dp).all():return fatal
     drank=int(np.linalg.lstsq(dp,np.zeros((n,1)),rcond=None)[2]);evidence['D_perp_rank']=drank;confirmed=drank<d.shape[1]
  if not confirmed:return fatal
  return {**fatal,'status':'identification_unavailable','reason':reason,'matrix_evidence':evidence}
 except (ValueError,TypeError,np.linalg.LinAlgError,FloatingPointError):return fatal
