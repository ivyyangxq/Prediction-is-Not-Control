"""Fixed prospective allocation. Import performs no science."""
CELLS=(('N','STABLE'),('CL','LATENT_F'),('CM','MODEL_A'),('MIX','MODEL_LATENT'))
SOURCE_BETA=.95
LIMIT={'constructors':348,'prefix_G':8700,'native_G':27840,'Q':13920,'reward':13920,'host_calls':25056,'transitions':801792}
THRESHOLDS={'scalar':.003925555745965009,'serial':.012974220288332923}
def streams():
    cal=[dict(index=4*i+c,phase='CAL',cell=cell,source=source,strength=1.,seed=3128400000+4*i+c,within_cell=i) for i in range(79) for c,(cell,source) in enumerate(CELLS)]
    ev=[dict(index=316+4*i+c,phase='EVAL',cell=cell,source=source,strength=1.,seed=3128400316+4*i+c,within_cell=i) for i in range(8) for c,(cell,source) in enumerate(CELLS)]
    return cal+ev
def dispatch_source(*,released_monitor_steps,monitor_available):return bool(monitor_available and released_monitor_steps==8)
def detector_flags(scores):
    flags={k:bool(scores[k]>v) for k,v in THRESHOLDS.items()}
    return dict(scores=scores,exceedances=flags,scope='old monitor thresholds metadata only; never select entry or inference')
