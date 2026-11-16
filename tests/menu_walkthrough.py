"""Actual ordered-variable/function/parameter menu walkthrough and guards."""
from pathlib import Path
import json
import numpy as np
from pinc_toolbox.product_ui import GQTool
from pinc_toolbox.tool.reward_wizard_v1 import install_reward_wizard, same_r0
from pinc_toolbox.tool.abcd_v1 import install_abcd
from pinc_toolbox.tool.product_v2 import launch_tool


def walk(root,input_directory,out):
    root=Path(root);input_directory=Path(input_directory);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    facade=launch_tool(root,show=False);ui=facade['ui'];reward=facade['reward'];abcd=facade['abcd']
    def button(label):
        def find(widget):
            if getattr(widget,'description',None)==label and hasattr(widget,'click'):return widget
            for child in getattr(widget,'children',[]):
                result=find(child)
                if result is not None:return result
            return None
        control=find(ui.root)
        if control is None:raise AssertionError('actual button missing: '+label)
        control.click()
    ui.example.value=False;ui.panel_path.value=str(input_directory/'panel.csv');ui.macro_path.value=str(input_directory/'public_proxy.csv');button('1 Load data and configure columns')
    for key,value in {'path_id':'unit','period':'date','release_period':'released','state':'x','action':'executed','next_state':'next_x','previous_action':'anchor'}.items():ui.columns[key].value=value
    ui.macro_scope.value='unit_proxy';ui.macro_key.value='unit';ui.macro_value.value='public_proxy';ui.macro_release.value='publication';button('2 Confirm mappings and timeline')
    for key,value in {'learn_start':0,'learn_end':24,'start':24,'end':47,'window':24,'warmup_as_of':8,'horizon':10,'initial_full':True,'retain_warmup_rows':True,'update_policy':'report_guided'}.items():ui.fields[key].value=value
    ui.grid.value=json.dumps(np.linspace(0,.1,11).tolist());ui.candidates.value=ui.grid.value
    initial=ui.history.loc[ui.history.period<24];ui.scale.value=json.dumps(np.maximum(initial[['state','w','previous_action','action']].std().to_numpy(),1e-8).tolist())
    ui.eta.value=str(json.loads((root/'experiments/full size experiment/FROZEN_GEOMETRIC_MEASUREMENT.json').read_text())['eta'])
    wizard=reward['wizard'];assert wizard is not None
    button('4 Create fixed-objective session');assert ui.session is None
    wizard.action_meaning.value='new_action_requires_adapter';wizard.variable_button.click();assert wizard.variables is None
    wizard.action_meaning.value='same_logged_action';wizard.variable_button.click();assert wizard.variables=={'x':'state','a':'action'}
    wizard.expression.value='x - curvature*x*x - cost*a - adjustment*a*a';wizard.form_button.click()
    assert set(wizard.parameters)=={'curvature','cost','adjustment'}
    wizard.ack.value=True;wizard.confirm_button.click();assert wizard.confirmed is None
    # A new unprovided state is rejected, rather than defaulted to a number.
    wizard.expression.value='x + hidden_state - cost*a';wizard.form_button.click()
    wizard.parameters['hidden_state']['kind'].value='unprovided_state'
    wizard.parameters['cost']['kind'].value='parameter';wizard.parameters['cost']['value'].value='2.025'
    wizard.ack.value=True;wizard.confirm_button.click();assert wizard.confirmed is None
    wizard.expression.value='x - curvature*x*x - cost*a - adjustment*a*a';wizard.form_button.click()
    for name,value in {'curvature':'.25','cost':'2.025','adjustment':'.5'}.items():
        wizard.parameters[name]['kind'].value='parameter';wizard.parameters[name]['value'].value=value
    wizard.ack.value=True;wizard.confirm_button.click();assert wizard.confirmed is not None
    spec=wizard.confirmed.copy();assert same_r0(spec)
    button('4 Create fixed-objective session');assert ui.session is not None
    assert ui.session.settings['reward']==spec
    for _ in range(16):button('Advance one period')
    ui.controlled_evidence_path.value=str(input_directory/'controlled_acquisition');button('Verify and attach controlled source evidence')
    assert ui.session.evidence is not None and ui.session.evidence.snapshot()['model_involvement']
    assert ui.session.settings['reward']==spec
    ui.repair_method.value='rolling_full_RA_GQ';button('Accept selected update after MODEL evidence');button('Run remaining periods')
    assert ui.session.next_date==48
    assert all(r['reward_digest']==ui.session.identity['reward'] for r in ui.session.timeline)
    assert ui.session.settings['reward']==spec
    ui.export_path.value=str(out/'ui_export');button('Export JSON and timeline CSV')
    summary={'status':'passed','actual_menu_order':['choose provided state/action roles and semantic interface','write function with selected aliases','classify and explicitly assign all numeric parameters','confirm fixed session objective'],
        'user_reward_unchanged':spec,'R0_equivalence':'exact polynomial normalization only for matching the old controlled CAL; user expression and parameters remain the actual learner reward',
        'guard_checks':['session creation before confirmation blocked','different new action without a response interface blocked','unassigned parameter blocked','unprovided state blocked'],
        'decisions':len(ui.session.timeline),'source':ui.session.evidence.snapshot()['source_outcome'],
        'repair':ui.session.repair,'geometric_eta':ui.session.settings['eta'],
        'single_entry_product_v2':True,
        'column_mapping_recorded':ui.history.attrs['column_mapping']['action']=='executed',
        'input_csv_hashes_recorded':all('sha256' in v for v in ui.history.attrs['input_files'].values()),
        'ABCD_menu_present':abcd['button'].description=='Execute all four ABCD paths',
        'scope':'isolated product-v2 menu engineering; formal frozen runtime unchanged'}
    (out/'ORDERED_REWARD_WALKTHROUGH.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (out/'ORDERED_REWARD_WALKTHROUGH.md').write_text('# Ordered reward menu walkthrough\n\n```json\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n```\n')
    return ui,reward,abcd,summary
