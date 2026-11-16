"""Optional ipywidgets front end. All numerical work lives in product.py."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .product import PanelSession, prepare_panel, demo_data, demo_settings
from .guided_gq import MODES


class GQTool:
    def __init__(self):
        import ipywidgets as w
        from IPython.display import display, Markdown
        self.w, self.display, self.Markdown = w, display, Markdown
        self.session = None
        self.output = w.Output()
        self.panel_path = w.Text(description='Panel CSV',layout=w.Layout(width='85%'))
        self.macro_path = w.Text(description='Macro CSV',layout=w.Layout(width='85%'))
        self.example = w.Checkbox(value=True,description='Use the built-in small example (not scientific evidence)')
        load = w.Button(description='1 Load data and configure columns',button_style='info')
        load.on_click(self._load)
        self.mapping_box=w.VBox()
        self.settings_box=w.VBox()
        self.evidence_box=w.VBox()
        self.root=w.VBox([w.HTML('<h3>GQ Tool: external panel and separately released public W</h3>'),self.example,
                         self.panel_path,self.macro_path,load,self.mapping_box,self.settings_box,
                         self.evidence_box,self.output])

    def show(self):
        self.display(self.root)
        return self

    def _safe(self, operation):
        with self.output:
            self.output.clear_output(wait=True)
            try:
                operation()
            except Exception as error:
                print(f'{type(error).__name__}: {error}')
                print('Correct the input and retry. Existing exports are preserved.')

    def _load(self, _=None):
        def op():
            if self.example.value:
                self.panel,self.macro,defaults=demo_data()
            else:
                self.panel=pd.read_csv(Path(self.panel_path.value).expanduser(),float_precision='round_trip')
                self.macro=pd.read_csv(Path(self.macro_path.value).expanduser(),float_precision='round_trip')
                defaults={}
            w=self.w
            labels={'path_id':'Unit ID','period':'Decision period','release_period':'Transition release period',
                    'state':'Current physical state','action':'Actual logged action','next_state':'Successor state',
                    'previous_action':'Previous actual action'}
            self.columns={key:w.Dropdown(options=([('Construct a within-unit lag',None)] if key=='previous_action' else [])+
                [(c,c) for c in self.panel.columns],value=defaults.get(key,None) if key=='previous_action' else
                defaults.get(key,self.panel.columns[0]),description=label,style={'description_width':'140px'},
                layout=w.Layout(width='90%')) for key,label in labels.items()}
            self.macro_value=w.Dropdown(options=list(self.macro),value='macro_W' if self.example.value else self.macro.columns[0],description='Public value')
            self.macro_release=w.Dropdown(options=list(self.macro),value='publication' if self.example.value else self.macro.columns[0],description='Publication period')
            self.macro_scope=w.Dropdown(options=[('Common public W','common'),('Unit-specific public proxy (not common W)','unit_proxy')],description='Public input scope',layout=w.Layout(width='95%'))
            self.macro_key=w.Dropdown(options=list(self.macro),description='Proxy unit key')
            configure=w.Button(description='2 Confirm mappings and timeline',button_style='info')
            configure.on_click(self._configure)
            self.mapping_box.children=[w.HTML('<b>Map columns</b>: actions are actual executed values. Distinguish common W from unit-specific public proxies; neither is a latent truth.'),*self.columns.values(),self.macro_value,self.macro_release,self.macro_scope,self.macro_key,configure]
            self.settings_box.children=[];self.evidence_box.children=[];self.session=None
            self.display(self.panel.head()); self.display(self.macro.head())
        self._safe(op)

    def _configure(self, _=None):
        def op():
            self.history=prepare_panel(self.panel,self.macro,{k:v.value for k,v in self.columns.items() if v.value is not None},
                                       self.macro_value.value,self.macro_release.value,
                                       macro_unit=self.macro_key.value if self.macro_scope.value=='unit_proxy' else None)
            w=self.w;s=demo_settings()
            periods=sorted(self.history.period.unique());middle=periods[len(periods)//2]
            example=self.example.value
            def number(key,label,value):
                return w.IntText(value=int(value),description=label,style={'description_width':'150px'})
            self.fields={
                'learn_start':number('learn_start','Learning start (inclusive)',s['learn_start'] if example else periods[0]),
                'learn_end':number('learn_end','Learning end (exclusive)',s['learn_end'] if example else middle),
                'start':number('start','Monitoring start (inclusive)',s['start'] if example else middle),
                'end':number('end','Monitoring end (inclusive)',s['end'] if example else periods[-1]),
                'window':number('window','Window (model periods)',24),
                'warmup_as_of':number('warmup_as_of','Full RA warmup',-24 if example else periods[0]+8),
                'horizon':number('horizon','Planning horizon',3),
                'path':w.Dropdown(options=[(label,pid) for pid,label in self.history.attrs['unit_labels'].items()],description='Query unit'),
                'discount':w.FloatText(value=.95,description='discount'),
                'mode':w.Dropdown(options=MODES,value=MODES[1],description='State input mode',layout=w.Layout(width='90%')),
                'initial_full':w.Checkbox(value=False,description='Use full auxiliary RA in historical learning'),
                'retain_warmup_rows':w.Checkbox(value=False,description='Retain neutral early warmup initial values (not measured residuals)'),
                'update_policy':w.Dropdown(options=[('Continuous rolling learning (not diagnosis-guided repair)','rolling'),('Fixed historical learning','fixed'),('User chooses an update after MODEL evidence','report_guided')],value='rolling',description='Update rule',layout=w.Layout(width='90%'))}
            # Use only the declared initial prefix to suggest the domain.
            prefix=self.history.loc[(self.history.period<self.fields['learn_end'].value)&(self.history.release_period<=self.fields['start'].value)]
            actions=sorted(prefix.action.dropna().unique())
            self.grid=w.Text(value=json.dumps(actions),description='Fixed action grid',layout=w.Layout(width='90%'))
            self.candidates=w.Text(value=json.dumps(actions),description='Candidate actions',layout=w.Layout(width='90%'))
            self.reward=w.Text(value=s['reward']['expression'],description='Fixed reward expression',layout=w.Layout(width='95%'))
            self.reward_inputs=w.Text(value=json.dumps(s['reward']['inputs']),description='Reward input mapping',layout=w.Layout(width='95%'))
            self.parameters=w.Text(value=json.dumps(s['reward']['parameters']),description='Fixed parameters JSON',layout=w.Layout(width='95%'))
            self.scale=w.Text(value=json.dumps(s['reference_scale']),description='Fixed reference scales',layout=w.Layout(width='95%'))
            self.eta=w.Text(value='null',description='Geometric eta',layout=w.Layout(width='95%'))
            self.reference=w.Text(value='null',description='Fixed public query',layout=w.Layout(width='95%'))
            init=w.Button(description='3 Create session',button_style='success');init.on_click(self._initialize)
            self.settings_box.children=[w.HTML('<b>Learning and updates</b>: periods below are integer model periods. Review the calendar mapping in the output. Map reward inputs to state / w / previous_action / action.'),
                *self.fields.values(),self.grid,self.candidates,self.reward,self.reward_inputs,self.parameters,
                w.HTML('Scale order: [state, W, previous_action, action]. eta=null means unspecified; distance measurement does not establish qualification. A null public query uses the first actual monitoring query.'),
                self.scale,self.eta,self.reference,init]
            self.display(self.history.head())
            if 'decision_calendar' in self.history.attrs:
                self.display(pd.DataFrame({'model_period':range(len(self.history.attrs['decision_calendar'])),'timestamp':self.history.attrs['decision_calendar']}))
            print('Available unit mapping:',self.history.attrs['unit_labels'])
        self._safe(op)

    def _initialize(self, _=None):
        def op():
            w=self.w
            s={key:field.value for key,field in self.fields.items()}
            s.update(grid=json.loads(self.grid.value),candidates=json.loads(self.candidates.value),
                     reward=dict(expression=self.reward.value,inputs=json.loads(self.reward_inputs.value),parameters=json.loads(self.parameters.value)),
                     reference_scale=json.loads(self.scale.value),eta=json.loads(self.eta.value))
            reference=json.loads(self.reference.value)
            if reference is not None:s['reference_query']=reference
            self.session=PanelSession(self.history,s)
            next_button=w.Button(description='Advance one period',button_style='info');next_button.on_click(lambda _:self._safe(lambda:self._show(self.session.advance())))
            all_button=w.Button(description='Run remaining periods',button_style='success');all_button.on_click(lambda _:self._safe(lambda:self._show(self.session.run())))
            report_button=w.Button(description='View unified report');report_button.on_click(lambda _:self._safe(lambda:self._show(self.session.report())))
            self.evidence_path=w.Text(description='Evidence request JSON',layout=w.Layout(width='95%'))
            self.controlled_evidence_path=w.Text(description='Matched controlled acquisition directory',layout=w.Layout(width='95%'))
            controlled_attach=w.Button(description='Verify and attach controlled source evidence');controlled_attach.on_click(self._attach_controlled)
            self.applicability=w.Textarea(value=json.dumps({'scope':'caller-declared finite-history construction','premises':['Supply actual identification, noise/MGF and source-family premises here'],'repair_applicable':False},ensure_ascii=False),description='Applicability JSON',layout=w.Layout(width='95%',height='100px'))
            attach=w.Button(description='Load evidence request');attach.on_click(self._attach)
            self.evidence_choice=w.Dropdown(options=[('Load a request first','')],description='Detection/source step',layout=w.Layout(width='95%'))
            evidence_step=w.Button(description='Execute selected evidence step');evidence_step.on_click(self._evidence_step)
            self.repair_method=w.Dropdown(options=['rolling_base_GQ','rolling_full_RA_GQ'],description='Repair method',layout=w.Layout(width='95%'))
            repair=w.Button(description='Accept selected update after MODEL evidence');repair.on_click(lambda _:self._safe(lambda:self._show(self.session.decide_repair(True,method=self.repair_method.value))))
            decline=w.Button(description='Decline diagnosis-guided update');decline.on_click(lambda _:self._safe(lambda:self._show(self.session.decide_repair(False))))
            self.export_path=w.Text(description='New export directory',layout=w.Layout(width='95%'))
            export=w.Button(description='Export JSON and timeline CSV');export.on_click(lambda _:self._safe(lambda:print(self.session.export(Path(self.export_path.value).expanduser()))))
            self.evidence_box.children=[w.HTML('<b>Execution and continuing support</b>: load evidence, then advance periods. Choose repair at the next unprocessed period.'),
                w.HBox([next_button,all_button,report_button]),w.HTML('<b>Optional certified detection/source workflow</b>: ordinary panel/W data do not automatically define the required finite model. This executes actual requests and rejects old MODEL labels. Missing evidence remains unavailable.'),
                self.controlled_evidence_path,controlled_attach,self.evidence_path,self.applicability,attach,self.evidence_choice,evidence_step,
                self.repair_method,w.HBox([repair,decline]),self.export_path,export]
            self._show(self.session.report())
        self._safe(op)

    def _choices(self):
        choices=self.session.evidence.snapshot()['native_snapshot']['choices']
        self.evidence_choice.options=[(c['label'],json.dumps([c['action'],c.get('node')])) for c in choices] or [('No further steps','')]

    def _attach_controlled(self,_=None):
        def op():
            from .controlled_product import evidence_for_panel
            self.session.evidence=evidence_for_panel(Path.cwd(),Path(self.controlled_evidence_path.value).expanduser(),self.session)
            self.session.source_reports.append(self.session.evidence.snapshot())
            self.evidence_choice.options=[('Controlled detection/source recomputed from actual records; no additional JSON step','')]
            self._show(self.session.evidence.snapshot())
        self._safe(op)

    def _attach(self,_=None):
        def op():
            request=json.loads(Path(self.evidence_path.value).expanduser().read_text())
            self._show(self.session.attach_evidence(request,applicability=json.loads(self.applicability.value)))
            self._choices()
        self._safe(op)

    def _evidence_step(self,_=None):
        def op():
            if not self.evidence_choice.value:raise ValueError('No executable step')
            action,node=json.loads(self.evidence_choice.value)
            self._show(self.session.evidence_step(action,node));self._choices()
        self._safe(op)

    def _show(self,value):
        if 'timeline' in value:
            columns=['date','status','fit_as_of','recorded_action','recommended_action','reward_value']
            if value['timeline']:self.display(pd.DataFrame(value['timeline'])[columns])
            print('Capabilities and limits:',json.dumps(value.get('capabilities',{}),ensure_ascii=False,indent=2))
            print(json.dumps({'identity':value.get('identity'),'rows':len(value['timeline']),
                'repair':value.get('repair'),'last_status':value['timeline'][-1]['status'] if value['timeline'] else None},ensure_ascii=False,indent=2))
            print('The session retains period-level data, support and source receipts. Export JSON/CSV for full details.')
        else:
            print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def controlled_benchmark_widget(root):
    """Optional research-checkout route: generate all diagnostic inputs internally."""
    import ipywidgets as w
    from IPython.display import display
    from .controlled_product import run_controlled_history
    family=w.Dropdown(options=['N','CL_Q100','CM','MIX'],description='Controlled scenario')
    seed=w.IntText(value=4239010011,description='Engineering seed')
    out=w.Text(description='New directory',layout=w.Layout(width='95%'))
    button=w.Button(description='Run one controlled-history smoke test',button_style='warning')
    output=w.Output()
    def run(_):
        with output:
            output.clear_output(wait=True)
            try:
                row={'ordinal':0,'candidate':family.value,'seed':seed.value,
                     'pair_group':0,'conditional_future_campaign_entropy':2026111300001}
                print('Generating fresh public history, actual H10 detection and reset source inputs; no old MODEL report is reused.')
                result=run_controlled_history(root,row,Path(out.value).expanduser())
                print(json.dumps(result,ensure_ascii=False,indent=2))
            except Exception as error:print(f'{type(error).__name__}: {error}')
    button.on_click(run)
    widget=w.VBox([w.HTML('<b>Optional controlled RND2 benchmark</b>: generates actual monitoring/source evidence without manual JSON. It uses path-specific proxies and common-innovation resets; it does not apply to ordinary common-W panels. It does not launch a full scientific batch.'),family,seed,out,button,output])
    display(widget)
    return widget
