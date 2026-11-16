"""Isolated ordered reward UI: select variables -> form -> explicit parameters.

No inference/estimation of reward, no new unobserved states, no historical
logged-action imputation. Install before the first GQTool load/configure click.
"""
from copy import deepcopy
from fractions import Fraction
import ast
import json
import keyword
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pinc_toolbox.reward_equation import RewardEquation
from pinc_toolbox.controlled_product import R0_REWARD, evidence_for_panel
from pinc_toolbox.product import digest

FUNCTIONS={'abs','exp','log','sqrt'}
ROLES=('state','w','previous_action','action')


def polynomial(spec):
    """Exact represented-constant polynomial normalization, no sampled matching."""
    inputs=spec['inputs'];parameters=spec['parameters'];zero=(0,0,0,0)
    def add(a,b):
        out=dict(a)
        for term,value in b.items():out[term]=out.get(term,Fraction(0))+value
        return {term:value for term,value in out.items() if value}
    def multiply(a,b):
        out={}
        for x,v in a.items():
            for y,w in b.items():
                term=tuple(i+j for i,j in zip(x,y));out[term]=out.get(term,Fraction(0))+v*w
        return {term:value for term,value in out.items() if value}
    def constant(value):return {zero:Fraction.from_float(float(value))}
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return constant(node.value)
        if isinstance(node,ast.Name):
            if node.id in parameters:return constant(parameters[node.id])
            role=inputs[node.id];power=[0]*4;power[ROLES.index(role)]=1;return {tuple(power):Fraction(1)}
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=visit(node.operand);return value if isinstance(node.op,ast.UAdd) else {k:-v for k,v in value.items()}
        if isinstance(node,ast.BinOp):
            a=visit(node.left)
            if isinstance(node.op,ast.Pow):
                if not isinstance(node.right,ast.Constant) or type(node.right.value) not in (int,float):raise ValueError('nonconstant power')
                power=int(node.right.value)
                if power!=node.right.value or not 0<=power<=8:raise ValueError('unsupported polynomial power')
                out=constant(1)
                for _ in range(power):out=multiply(out,a)
                return out
            b=visit(node.right)
            if isinstance(node.op,ast.Add):return add(a,b)
            if isinstance(node.op,ast.Sub):return add(a,{k:-v for k,v in b.items()})
            if isinstance(node.op,ast.Mult):return multiply(a,b)
            if isinstance(node.op,ast.Div) and set(b)=={zero} and b[zero]:return {k:v/b[zero] for k,v in a.items()}
        raise ValueError('not an exact supported polynomial representation')
    return visit(ast.parse(spec['expression'],mode='eval').body)


def same_r0(spec):
    try:return polynomial(spec)==polynomial(R0_REWARD)
    except (ValueError,KeyError,TypeError,ZeroDivisionError,OverflowError):return False


def attach_equivalent_controlled_evidence(root,directory,session):
    if not same_r0(session.settings['reward']):
        raise ValueError('This acquisition uses calibration for fixed RND2 R0. The current reward differs, so this source route does not apply. Other applications need matching valid measurement and calibration.')
    # Only the validation fingerprint is normalized; the user's actual reward
    # form/parameter values and the learner's equation remain untouched.
    identity={**session.identity,'reward':digest(R0_REWARD)}
    evidence=evidence_for_panel(root,directory,SimpleNamespace(history=session.history,identity=identity))
    evidence.binding=deepcopy(session.identity)
    evidence.request['reward_equivalence']='exact polynomial equality to calibrated RND2 R0; actual user form/parameters retained'
    evidence.applicability['premises'].append('user-declared reward exactly matches calibrated R0 after represented-constant polynomial normalization')
    return evidence


class RewardWizard:
    def __init__(self,ui):
        self.ui=ui;w=ui.w;self.confirmed=None;self.variables=None;self.parameters={}
        self.output=w.Output();self.boxes={};self.aliases={}
        labels={'state':'Provided physical column '+str(ui.columns['state'].value),
                'w':('Provided public proxy ' if ui.macro_scope.value=='unit_proxy' else 'Provided common W ')+str(ui.macro_value.value),
                'previous_action':'Previous actual action (provided column or valid within-unit lag)',
                'action':'Choice/action to optimize (distinct from historical logged actions)'}
        defaults={'state':'x','w':'W','previous_action':'p','action':'a'}
        rows=[]
        for role in ROLES:
            self.boxes[role]=w.Checkbox(value=role in ('state','action'),description=labels[role],layout=w.Layout(width='65%'))
            self.aliases[role]=w.Text(value=defaults[role],description='Expression alias',layout=w.Layout(width='35%'))
            rows.append(w.HBox([self.boxes[role],self.aliases[role]]))
        self.action_meaning=w.Dropdown(options=[('Confirm the economic meaning of action',None),
            ('Same action as the mapped historical actual action; optimize candidate values','same_logged_action'),
            ('A different new action without a matching response interface','new_action_requires_adapter')],value=None,
            description='Action interface',layout=w.Layout(width='98%'))
        self.variable_button=w.Button(description='3a Confirm selected inputs and choice',button_style='info')
        self.expression=w.Textarea(value='',placeholder='Use only confirmed aliases, e.g. x - curvature*x*x - cost*a - adjustment*a*a',
            description='Reward function',disabled=True,layout=w.Layout(width='98%',height='80px'))
        self.form_button=w.Button(description='3b Confirm function form',disabled=True,button_style='info')
        self.parameter_box=w.VBox()
        self.ack=w.Checkbox(value=False,description='Confirm: these are fixed numerical parameters, not missing data/state. Keep this form and parameters fixed throughout the session.',disabled=True,layout=w.Layout(width='98%'))
        self.confirm_button=w.Button(description='3c Confirm all parameters and fix reward',disabled=True,button_style='success')
        self.variable_button.on_click(self.choose_variables);self.form_button.on_click(self.choose_form);self.confirm_button.on_click(self.confirm)
        self.expression.observe(self.form_changed,names='value')
        for widget in [*self.boxes.values(),*self.aliases.values(),self.action_meaning]:widget.observe(self.variables_changed,names='value')
        self.panel=w.VBox([w.HTML('<b>Set the reward in order: provided inputs / choice, function form, then every numerical parameter.</b><br>Do not invent unprovided states. Q/value may optimize a choice; it is not the current observed value. Do not fill missing historical actual actions with recommendations. A different action requires a new transition/response identification interface.'),
            *rows,self.action_meaning,self.variable_button,self.expression,self.form_button,self.parameter_box,self.ack,self.confirm_button,self.output])

    def say(self,message):
        with self.output:self.output.clear_output(wait=True);print(message)

    def variables_changed(self,_=None):
        self.variables=None;self.confirmed=None;self.expression.disabled=True;self.form_button.disabled=True
        self.confirm_button.disabled=True;self.ack.disabled=True;self.ack.value=False
        self.parameter_box.children=[];self.parameters={}

    def form_changed(self,_=None):
        self.confirmed=None;self.confirm_button.disabled=True;self.ack.disabled=True;self.ack.value=False
        self.parameter_box.children=[];self.parameters={}

    def choose_variables(self,_=None):
        try:
            selected={}
            for role in ROLES:
                if not self.boxes[role].value:continue
                alias=self.aliases[role].value.strip()
                if not alias.isidentifier() or keyword.iskeyword(alias) or alias in FUNCTIONS:raise ValueError('Aliases must be valid identifiers and must not shadow allowed functions')
                if alias in selected:raise ValueError('Every input and choice needs a distinct alias')
                selected[alias]=role
            if not selected:raise ValueError('Select at least one provided input or choice first')
            if self.boxes['action'].value and self.action_meaning.value!='same_logged_action':
                raise ValueError('Confirm that this optimizes values of the same historical action. A new action requires an explicit transition/response identification interface; the native adapter does not assume identical effects.')
            self.variables=selected;self.expression.disabled=False;self.form_button.disabled=False
            self.say('Confirmed:'+json.dumps(selected,ensure_ascii=False)+'. Write the function using these aliases; the reward is not fixed yet.')
        except Exception as error:self.say(str(error))

    def choose_form(self,_=None):
        try:
            if self.variables is None:raise ValueError('Complete step 3a first')
            expression=self.expression.value.strip()
            if not expression:raise ValueError('Enter the reward function form')
            tree=ast.parse(expression,mode='eval');names={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}-FUNCTIONS
            unselected={self.aliases[r].value.strip() for r in ROLES if not self.boxes[r].value}
            if names&unselected:raise ValueError('The function uses an unselected data input:'+', '.join(sorted(names&unselected))+'; select it in step 3a first')
            parameters=sorted(names-set(self.variables))
            RewardEquation(expression,self.variables,{name:1. for name in parameters})
            self.parameters={};rows=[];w=self.ui.w
            for name in parameters:
                kind=w.Dropdown(options=[('Declare the symbol type',None),('Fixed numerical parameter','parameter'),('Unprovided data/state (unavailable)','unprovided_state')],value=None,description=name,layout=w.Layout(width='68%'))
                value=w.Text(value='',placeholder='Enter an explicit finite numerical value',layout=w.Layout(width='32%'))
                self.parameters[name]={'kind':kind,'value':value};rows.append(w.HBox([kind,value]))
                kind.observe(self.parameter_changed,names='value');value.observe(self.parameter_changed,names='value')
            self.parameter_box.children=rows or [w.HTML('No unassigned symbolic parameters remain. Numerical constants in the expression are explicitly fixed values.')]
            self.ack.disabled=False;self.confirm_button.disabled=False
            self.say('Function form confirmed. Classify and assign every numerical parameter. If a symbol is missing state, supply data or a valid prediction; do not disguise it as a parameter.')
        except Exception as error:self.say(str(error))

    def parameter_changed(self,_=None):self.confirmed=None;self.ack.value=False

    def confirm(self,_=None):
        try:
            if self.variables is None or self.confirm_button.disabled:raise ValueError('Complete steps 3a and 3b in order first')
            values={}
            for name,widgets in self.parameters.items():
                if widgets['kind'].value!='parameter':raise ValueError(name+' must be declared a fixed numerical parameter; unprovided states cannot be added')
                text=widgets['value'].value.strip()
                if not text:raise ValueError(name+' has no assigned value')
                try:value=float(text)
                except ValueError:raise ValueError(name+' requires a numerical value')
                if not np.isfinite(value):raise ValueError(name+' must be finite')
                values[name]=value
            if not self.ack.value:raise ValueError('Confirm the function and parameter meanings and their fixed use throughout this session')
            spec={'expression':self.expression.value.strip(),'inputs':deepcopy(self.variables),'parameters':values}
            RewardEquation(spec['expression'],spec['inputs'],spec['parameters'])
            self.confirmed=deepcopy(spec)
            self.say('Reward fixed; the tool will not estimate a different reward:\n'+json.dumps(spec,ensure_ascii=False,indent=2)+'\nYou may now create a session. Later edits do not change an existing session objective.')
        except Exception as error:self.say(str(error))


def install_reward_wizard(ui,root):
    """Wrap the existing UI callbacks without changing the frozen source files."""
    root=Path(root);original_configure=ui._configure;original_initialize=ui._initialize
    state={'wizard':None}
    def configure(_=None):
        before=id(getattr(ui,'history',None));original_configure(_)
        if id(getattr(ui,'history',None))==before:return
        wizard=RewardWizard(ui);state['wizard']=wizard
        children=list(ui.settings_box.children)
        hidden={id(ui.reward),id(ui.reward_inputs),id(ui.parameters)}
        children=[c for c in children if id(c) not in hidden]
        for i,child in enumerate(children):
            if getattr(child,'description',None)=='3 Create session':
                child.description='4 Create fixed-objective session';children.insert(i,wizard.panel);break
        ui.settings_box.children=children
    def initialize(_=None):
        wizard=state['wizard']
        if wizard is None or wizard.confirmed is None or not wizard.ack.value:
            ui._safe(lambda: (_ for _ in ()).throw(ValueError('Select inputs and choice, enter a function, explicitly assign every parameter and confirm before creating a session.')))
            return
        spec=deepcopy(wizard.confirmed)
        ui.reward.value=spec['expression'];ui.reward_inputs.value=json.dumps(spec['inputs']);ui.parameters.value=json.dumps(spec['parameters'])
        ui.fields['reward_declaration']=SimpleNamespace(value={'ordered_steps':['provided variables and declared choice','user function form','all numeric parameter values explicit'],
            'specification':deepcopy(spec),'action_semantics':'same meaning as historical logged action; hypothetical candidate values are optimized, never substituted into history',
            'unprovided_states':'not allowed; requires actual provided data or a lawful existing prediction'})
        original_initialize(_)
    def controlled_attach(_=None):
        def operation():
            ui.session.evidence=attach_equivalent_controlled_evidence(root,Path(ui.controlled_evidence_path.value).expanduser(),ui.session)
            ui.session.source_reports.append(ui.session.evidence.snapshot())
            ui.evidence_choice.options=[('Controlled source evidence verified; the user function and parameters remain unchanged','')]
            ui._show(ui.session.evidence.snapshot())
        ui._safe(operation)
    ui._configure=configure;ui._initialize=initialize;ui._attach_controlled=controlled_attach
    return state
