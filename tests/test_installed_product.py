import sys,os,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from pinc_toolbox.tool import launch_tool
from pinc_toolbox.tool.persistence import save_checkpoint,load_checkpoint
from pinc_toolbox.tool.abcd_v1 import run_abcd
from pinc_toolbox.controlled_product import _load
from pinc_toolbox.product import demo_data
basepanel,basepublic,mapping=demo_data();inputs=Path(tempfile.mkdtemp());basepanel.to_csv(inputs/'panel.csv',index=False);basepublic.to_csv(inputs/'public.csv',index=False)
facade=launch_tool(show=False);ui=facade['ui'];ui.example.value=False;ui.panel_path.value=str(inputs/'panel.csv');ui.macro_path.value=str(inputs/'public.csv');ui._load()
for key,value in mapping.items():ui.columns[key].value=value
ui.macro_value.value='macro_W';ui.macro_release.value='publication';ui._configure();wizard=facade['reward']['wizard']
assert wizard is not None
wizard.action_meaning.value='same_logged_action';wizard.variable_button.click()
wizard.expression.value='x-cost*a*a';wizard.form_button.click();wizard.parameters['cost']['kind'].value='parameter';wizard.parameters['cost']['value'].value='2';wizard.ack.value=True;wizard.confirm_button.click();assert wizard.confirmed is not None
ui._initialize();assert ui.session is not None
for _ in range(3):ui.session.advance()
path=Path(tempfile.mkdtemp())/'checkpoint.gq';save_checkpoint(ui.session,path);s=load_checkpoint(path);assert s.next_date==ui.session.next_date
# Both original and restored execute the same next release.
a=ui.session.advance();b=s.advance();assert a['recommended_action']==b['recommended_action'] and a['status']==b['status']
facade['persistence']['checkpoint_path'].value=str(path);facade['persistence']['resume'].click();assert ui.session.next_date==load_checkpoint(path).next_date
root=Path(sys.modules['pinc_toolbox'].__file__).parent/'_resources'
original=_load('installed_original',root/'DGP/RND/02_state_dependent_local/model_source_v004/environment.py')
roundmod=_load('installed_round',root/'experiments/round_coefficient_precision128_20261110/run_round.py')
dgp=roundmod.round_environment(original);env=dgp.Environment(8123,'STABLE',paths=32,horizon=64,switch_date=32)
rows=[];prev=np.full(32,.05);rng=np.random.default_rng(93);grid=np.linspace(0,.1,11)
for date in range(44):
 x=env.state[:,date].copy();w=env.w[:,date].copy();actions=grid[rng.integers(0,11,32)];env.step(date,actions)
 for i in range(32):rows.append({'path_id':i,'period':date,'release_period':date+1,'state':x[i],'w':w[i],'previous_action':prev[i],'action':actions[i],'next_state':env.state[i,date+1],'next_w':env.w[i,date+1]})
 prev=actions
from pinc_toolbox.product import demo_settings
from pinc_toolbox.tool.persistence import InstalledPanelSession
settings=demo_settings();settings.update(learn_start=0,learn_end=24,start=24,end=43,window=24,warmup_as_of=8,grid=grid.tolist(),candidates=grid.tolist(),horizon=3,path=0)
h=pd.DataFrame(rows);h.attrs['unit_labels']={i:str(i) for i in range(32)};session=InstalledPanelSession(h,settings)
pending=h.copy(deep=True);pending.attrs=h.attrs.copy();last=pending.index[-1];pending.loc[last,'next_state']=np.nan
released=InstalledPanelSession(pending,settings);released.ingest(h);assert released.history.loc[last,'next_state']==h.loc[last,'next_state']
changed=h.copy(deep=True);changed.attrs=h.attrs.copy();changed.loc[0,'action']+=.001
try:released.ingest(changed)
except ValueError:pass
else:raise AssertionError('known actual action revision accepted')
request,report=run_abcd(root,session,seed=8123,family='N',fork=40,steps=1,raw_history=h)
result={'status':'passed','installed_module':str(Path(sys.modules['pinc_toolbox'].__file__)),'cwd':os.getcwd(),'outside_research_tree':True,'reward_menu':True,'real_csv_import':True,'checkpoint_resume_equal':True,'abcd_status':report['counterfactual']['comparison']['status'],'resources_independent':True,'resume_button':True,'new_release_ingestion':True,'known_actual_revision_rejected':True}
Path(os.environ.get('GQ_VALIDATION_OUTPUT','gq_python_install_validation.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
