"""Installed one-seed test: python test_one_seed_end_to_end.py --output NEW_DIR."""
import argparse,json,time
from pathlib import Path
import pandas as pd
import pinc_toolbox
from pinc_toolbox.controlled_product import fresh_round_monitor
from pinc_toolbox.tool.persistence import save_checkpoint,load_checkpoint
from pinc_toolbox.tool.abcd_v1 import run_abcd
from menu_walkthrough import walk

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.time()
 root=Path(pinc_toolbox.__file__).parent/'_resources';assignment=dict(ordinal=2,seed=3260040000,candidate='CM',conditional_future_campaign_entropy=202610040000,pair_group=0)
 print('Fresh actual acquisition',assignment,flush=True);source,raw=fresh_round_monitor(root,assignment,out/'controlled_acquisition',end=63)
 print('Fresh source',source['source_outcome'],'ALARM',source['stage1']['alarm'],flush=True)
 panel=raw.rename(columns={'path_id':'unit','period':'date','release_period':'released','state':'x','action':'executed','next_state':'next_x','previous_action':'anchor'})[['unit','date','released','x','executed','next_x','anchor']]
 public=pd.concat([raw[['path_id','period','w']].rename(columns={'path_id':'unit','period':'publication','w':'public_proxy'}),raw.loc[raw.period==63,['path_id','release_period','next_w']].rename(columns={'path_id':'unit','release_period':'publication','next_w':'public_proxy'})]).drop_duplicates(['unit','publication'])
 panel.to_csv(out/'panel.csv',index=False,float_format='%.17g');public.to_csv(out/'public_proxy.csv',index=False,float_format='%.17g')
 ui,reward,controls,result=walk(root,out,out/'actual_menu_run')
 save_checkpoint(ui.session,out/'checkpoint.gq');s=load_checkpoint(out/'checkpoint.gq');assert s.next_date==ui.session.next_date and s.report()==ui.session.report()
 _,abcd=run_abcd(root,ui.session,seed=assignment['seed'],family='CM',fork=40,steps=2,raw_history=ui.session.history)
 (out/'ABCD.json').write_text(json.dumps(abcd,indent=2,allow_nan=False))
 result.update(assignment=assignment,installed_module=str(Path(pinc_toolbox.__file__).resolve()),fresh_acquisition=True,checkpoint_equal=True,abcd_comparison=abcd['counterfactual']['comparison'],elapsed_seconds=time.time()-start,scope='One-seed installed integration test; not power, size or economic recovery evidence.')
 (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
