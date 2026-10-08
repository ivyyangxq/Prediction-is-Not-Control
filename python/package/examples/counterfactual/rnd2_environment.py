"""Explicit small public-w simulator, not a truth input to the RA learner.

w is an observed boundary condition, not latent Z. The adapter replaces only
w at each decision; state and previous action remain endogenous. Outcomes
retain that period's w; the next dated boundary condition arrives separately.
The public w series is predictable by design in this illustration. Innovations
are private period-indexed simulator inputs; no outcome is delivered pre-action.
"""
import numpy as np
from pinc_toolbox.counterfactual import PublicInput, PublicOutcome


class RND2PublicEnvironment:
    public_input_names=('w',)
    def __init__(self, initial_w, drift=.004, seed=812):
        self.initial_w=float(initial_w);self.drift=float(drift);self.seed=int(seed)
    def public_input(self,time):
        return PublicInput(np.array([self.initial_w+self.drift*time]),time)
    def condition(self,state,public):
        state.physical[1]=public.values[0]
        return state
    def step(self,time,state,action,public):
        x,w,prev=state.physical
        innovation=np.random.default_rng(np.random.SeedSequence([self.seed,time])).normal(0,.006)
        nxt=.3+.4*x+.2*w+.1*prev+.5*(action-prev)+innovation
        reward=1-.3*x*x+2*action-30*action*action+10*w*action
        return PublicOutcome(np.array([nxt,w,action]),float(reward),time+1)


def create_environment(config):
    return RND2PublicEnvironment(**config)
