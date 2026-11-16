"""Installed-only checkpoint and released-file ingestion; scientific core unchanged."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,cloudpickle
import numpy as np,pandas as pd
from pinc_toolbox.product import PanelSession,frame_digest

class InstalledPanelSession(PanelSession):
    def ingest(self,history):
        if self.history.attrs.get('unit_labels')!=history.attrs.get('unit_labels'):
            raise ValueError('Keep the existing unit-label mapping; include the full known-unit file')
        keys=['path_id','period'];old=self.history.set_index(keys);new=history.set_index(keys)
        if new.index.has_duplicates:raise ValueError('duplicate unit/date')
        for key in old.index.intersection(new.index):
            a,b=old.loc[key],new.loc[key]
            for col in old.columns.intersection(new.columns):
                x,y=a[col],b[col]
                if pd.isna(x) and pd.notna(y):
                    if a.release_period<self.next_date:raise ValueError('Cannot revise already processed released records')
                elif not ((pd.isna(x) and pd.isna(y)) or x==y):
                    raise ValueError(f'Existing record changed: {key}, {col}')
        combined=new.combine_first(old).reset_index().sort_values(keys,kind='stable').reset_index(drop=True)
        combined.attrs=deepcopy(history.attrs)
        previous=deepcopy(self.identity);self.history=combined;self.identity['history']=frame_digest(combined)
        self.ingestion_log=getattr(self,'ingestion_log',[])+[{'previous_identity':previous,'identity':deepcopy(self.identity),'next_date':self.next_date}]
        self.evidence=None # A previous source binding cannot authorize newly supplied data.
        self.ra_cache=None
        return {'status':'ingested','rows':len(combined),'next_date':self.next_date,'source_binding':'fresh evidence required'}

def save_checkpoint(session,path):
    path=Path(path)
    if path.exists():raise ValueError('Choose a new checkpoint path')
    payload=cloudpickle.dumps(session,protocol=5)
    envelope={'format':'gq-tool-native-python-1','sha256':hashlib.sha256(payload).hexdigest(),'payload':payload}
    path.write_bytes(cloudpickle.dumps(envelope,protocol=5));return str(path)

def load_checkpoint(path):
    # Checkpoints are user-owned Python objects, not external evidence reports.
    envelope=cloudpickle.loads(Path(path).read_bytes())
    if envelope['format']!='gq-tool-native-python-1' or hashlib.sha256(envelope['payload']).hexdigest()!=envelope['sha256']:raise ValueError('Checkpoint integrity mismatch')
    session=cloudpickle.loads(envelope['payload'])
    if not isinstance(session,InstalledPanelSession):raise ValueError('Installed session checkpoint required')
    return session

def install(ui):
    import pinc_toolbox.product_ui as module
    module.PanelSession=InstalledPanelSession
    w=ui.w;path=w.Text(description='Checkpoint',layout=w.Layout(width='95%'))
    save=w.Button(description='Save resumable checkpoint');resume=w.Button(description='Restore checkpoint')
    ingest=w.Button(description='Ingest updated CSV / new releases')
    def saved(_):ui._safe(lambda:print(save_checkpoint(ui.session,path.value)))
    def restored(_):
        def op():
            session=load_checkpoint(path.value);s=session.settings;ui.history=session.history
            ui.fields={k:SimpleNamespace(value=v) for k,v in s.items()}
            for attr,value in [('grid',s['grid']),('candidates',s['candidates']),('reward',s['reward']['expression']),('reward_inputs',s['reward']['inputs']),('parameters',s['reward']['parameters']),('scale',s['reference_scale']),('eta',s.get('eta')),('reference',s.get('reference_query'))]:
                setattr(ui,attr,SimpleNamespace(value=value if attr=='reward' else json.dumps(value)))
            module.GQTool._initialize(ui);ui.session=session;ui._show(session.report())
        ui._safe(op)
    def ingested(_):
        def op():
            from pinc_toolbox.product import prepare_panel
            panel=pd.read_csv(Path(ui.panel_path.value).expanduser(),float_precision='round_trip');public=pd.read_csv(Path(ui.macro_path.value).expanduser(),float_precision='round_trip')
            attrs=ui.session.history.attrs;mapping=attrs['column_mapping'];public_map=attrs['public_series_mapping']
            history=prepare_panel(panel,public,mapping,public_map['value'],public_map['publication'],macro_unit=public_map['unit_key'])
            history.attrs.update(column_mapping=mapping,public_series_mapping=public_map)
            print(ui.session.ingest(history));ui.history=ui.session.history
        ui._safe(op)
    save.on_click(saved);resume.on_click(restored);ingest.on_click(ingested)
    panel=w.VBox([w.HTML('<b>Installed edition: resume and new releases</b>'),path,w.HBox([save,resume]),ingest])
    ui.root.children=(*ui.root.children[:-1],panel,ui.output)
    return {'panel':panel,'checkpoint_path':path,'save':save,'resume':resume,'ingest':ingest}
