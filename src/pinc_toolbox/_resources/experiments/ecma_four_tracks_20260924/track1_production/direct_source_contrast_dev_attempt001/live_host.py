"""Trusted oracle host using the original model_source_v004 Environment.

The adapter performs no structural reimplementation and sends no truth to learner.
"""
from copy import deepcopy
import numpy as np
class OriginalEnvironmentHost:
    def __init__(self,environment):self._env=deepcopy(environment)
    def public_main_proxy(self,time):
        if time!=self._env._next_t:raise ValueError('host public clock mismatch')
        return float(self._env.public(time)[1][0])
    def step_panel(self,time,actual_physical,actions):
        if time!=self._env._next_t:raise ValueError('nonsequential host step')
        x,w=self._env.public(time)
        previous=self._env.action[:,time-1] if time else np.full(32,.05)
        expected=np.column_stack([x,w,previous])
        if not np.array_equal(actual_physical,expected):raise ValueError('host physical state differs from actual released state')
        # Delegates to the original full source dispatch, including MODEL_LATENT.
        self._env.step(time,np.asarray(actions,float))
        nx,nw=self._env.public(time+1)
        return np.column_stack([nx,nw,actions]),self._env.released_reward(time+1),time+1
