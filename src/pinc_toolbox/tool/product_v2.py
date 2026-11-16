"""One-entry isolated product facade, ready for the final main notebook merge."""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pinc_toolbox.product_ui import GQTool
from .reward_wizard_v1 import install_reward_wizard
from .abcd_v1 import install_abcd


def _sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def install_geometry_and_provenance(ui,root):
    root=Path(root);original=ui._configure;state={}
    def configure(_=None):
        before=id(getattr(ui,'history',None));original(_)
        if id(getattr(ui,'history',None))==before:return
        ui.history.attrs['column_mapping']={key:control.value for key,control in ui.columns.items()}
        ui.history.attrs['public_series_mapping']={'value':ui.macro_value.value,'publication':ui.macro_release.value,
            'unit_key':ui.macro_key.value if ui.macro_scope.value=='unit_proxy' else None,'scope':ui.macro_scope.value}
        if ui.example.value:
            ui.history.attrs['input_files']={'source':'built-in engineering example; not scientific evidence'}
        else:
            ui.history.attrs['input_files']={key:{'path':str(Path(path).expanduser().resolve()),'sha256':_sha_file(Path(path).expanduser())}
                for key,path in [('panel',ui.panel_path.value),('public_series',ui.macro_path.value)]}
        w=ui.w
        choice=w.Dropdown(options=[('Specify eta manually (null if unspecified)','manual'),
            ('Compute leave-one-out coverage from the learning prefix and fixed scales','prefix'),
            ('Use the fixed independent RND2 engineering-prefix preset','rnd2_preset')],value='manual',description='Geometric eta source',layout=w.Layout(width='98%'))
        scales=w.Button(description='Set fixed scales from learning prefix')
        calculate=w.Button(description='Set fixed geometric eta')
        output=w.Output()
        def prefix():
            if ui.session is not None:raise ValueError('Set fixed scales and eta before creating a session. Do not widen them during execution to restore support.')
            first=max(ui.fields['learn_start'].value,ui.fields['learn_end'].value-ui.fields['window'].value)
            table=ui.history.loc[(ui.history.period>=first)&(ui.history.period<ui.fields['learn_end'].value)&(ui.history.release_period<=ui.fields['start'].value)]
            columns=['state','w','previous_action','action','next_state','next_w']
            table=table.loc[np.isfinite(table[columns].to_numpy(float)).all(axis=1)]
            if len(table)<2:raise ValueError('At least two complete released joint records are required. Missing values are not zero-filled.')
            return table
        def run(operation):
            with output:
                output.clear_output(wait=True)
                try:operation()
                except Exception as error:print(str(error))
        def scale_op():
            table=prefix();values=np.maximum(table[['state','w','previous_action','action']].std().to_numpy(float),1e-8)
            ui.scale.value=json.dumps(values.tolist());print('Reference scales set from the initial prefix; they remain fixed after session creation.')
        def eta_op():
            table=prefix()
            if choice.value=='manual':print('Use the manual eta field. null explicitly means unspecified.');return
            if choice.value=='rnd2_preset':
                config=json.loads((root/'experiments/full size experiment/FROZEN_GEOMETRIC_MEASUREMENT.json').read_text());eta=config['eta']
                source='independent RND2 engineering prefix preset; descriptive geometry only'
            else:
                scale=np.asarray(json.loads(ui.scale.value),float)
                if scale.shape!=(4,) or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Specify four positive fixed reference scales first')
                points=table[['state','w','previous_action','action']].to_numpy(float)/scale
                distances,_=NearestNeighbors(n_neighbors=2,metric='chebyshev').fit(points).kneighbors(points)
                eta=float(distances[:,1].max());source='initial learning prefix leave-one-out maximum scaled L-infinity distance'
            ui.eta.value=json.dumps(eta)
            ui.history.attrs['geometric_measurement_choice']={'eta':eta,'source':source,
                'scope':'measurement only; native runtime support and source/repair rules unchanged','prefix_rows':len(table)}
            print('Fixed geometric eta set to',eta,'. This is not an error certificate and does not change the native controller.')
        scales.on_click(lambda _:run(scale_op));calculate.on_click(lambda _:run(eta_op))
        panel=w.VBox([w.HTML('<b>Fixed geometric measurement</b>: use only the learning prefix or explicitly declared values. Keep it separate from native support, action memory, Bellman finiteness and economic value qualification.'),choice,w.HBox([scales,calculate]),output])
        children=list(ui.settings_box.children)
        index=next((i for i,c in enumerate(children) if getattr(c,'description',None)=='4 Create fixed-objective session'),len(children))
        children.insert(index,panel);ui.settings_box.children=children
        state.update(choice=choice,scales=scales,calculate=calculate,panel=panel)
    ui._configure=configure
    return state


def launch_tool(root=None,*,show=True):
    root=Path(__file__).resolve().parents[1]/'_resources' if root is None else Path(root)
    ui=GQTool()
    from .persistence import install
    persistence=install(ui)
    reward=install_reward_wizard(ui,root)
    geometry=install_geometry_and_provenance(ui,root);abcd=install_abcd(ui,root)
    if show:ui.show()
    return {'persistence':persistence,'ui':ui,'reward':reward,'geometry':geometry,'abcd':abcd,
            'version':'product-v2'}
