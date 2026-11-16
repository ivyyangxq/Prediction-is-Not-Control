"""Predeclared eight-history DEV; importing does not execute science."""
from pathlib import Path
CELLS=(('N','STABLE'),('CL','LATENT_F'),('CM','MODEL_A'),('MIX','MODEL_LATENT'))
ACTIONS=(.02,.08)
LIMIT={'constructors':8,'prefix_G':200,'native_G':256,'Q':128,'reward':128,'host_calls':512,'transitions':16384}
OUTPUT_BYTES=100*1024**2
TIME_SECONDS=1800
def streams():
 return [dict(index=i,phase='DEV',cell=CELLS[i%4][0],source=CELLS[i%4][1],strength=1.,seed=3128500000+i,within_cell=i//4) for i in range(8)]
def configuration():
 return dict(plan=streams(),actions=list(ACTIONS),paths=32,prefix_steps=32,monitor_dates=list(range(32,40)),source_dates=list(range(40,48)),warmup_as_of=8,window=24,horizon=10,beta=.95,nodes_per_memory=5,noise_nodes=4,switch_date=32,profile='step',limits=LIMIT,output_limit_bytes=OUTPUT_BYTES,time_limit_seconds=TIME_SECONDS,free_space_reserve_bytes=512*1024**2,planned_probe_endpoints=128,planned_pair_dates=64,science_execution_authorization='pending_Root_after_ordinary_review',scope='same-date reset probes; not continuous branch policies or source calibration',new_thresholds=False,new_p_values=False,eta_status='not_supplied')
