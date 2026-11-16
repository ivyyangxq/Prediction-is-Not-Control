"""Restore a trusted local public learner snapshot; never prepare or generate.

The archive locator belongs to provenance, not to posterior filter inputs.
Only the target learner payload is deserialized; prior snapshots supply public
history deltas. No environment object is restored. The caller supplies the
original Panel class, avoiding scientific-runner import side effects here.
"""
import copy
import json
import lzma
import pickle
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd


def restore_boundary(archive, boundary_path, panel_class):
    boundary = json.loads(Path(boundary_path).read_text())
    if boundary['release_time'] != 40 or boundary['pending_issue'] is not False:
        raise ValueError('expected released boundary40 without pending issue')
    previous = {}
    released = {}
    with lzma.open(archive, 'rb') as stream:
        while True:
            record = pickle.load(stream)
            if record[0] == 'end':
                raise ValueError('released learner40 absent')
            if record[0] == 'raw' and record[1] in ('RELEASED_40.json', 'AUXILIARY_40.json'):
                if record[1] in released:
                    raise ValueError('duplicate boundary release record')
                released[record[1]] = json.loads(record[2])
            if record[0] != 'snapshot':
                continue
            _, name, chain, base, delta, payload = record
            history = delta.copy(deep=True) if base is None else pd.concat(
                [previous[base], delta], ignore_index=True)
            previous[chain] = history
            if name != 'released_learner_40.pkl.gz':
                continue
            learner = cloudpickle.loads(payload)
            learner.history = history.copy(deep=True)
            break
    if learner.pending is not None or learner.model.as_of_ != 39:
        raise ValueError('unexpected learner pending state or fitted vintage')
    if (len(history) != 1024 or history.period.min() != 8 or
            history.period.max() != 39 or history.release_period.max() != 40):
        raise ValueError('unexpected public learner history boundary')
    physical = np.asarray(boundary['panel_physical'], dtype=float)
    auxiliary = np.asarray(boundary['panel_auxiliary'], dtype=float)
    if any(v.shape != (32, 3) or not np.isfinite(v).all()
           for v in (physical, auxiliary)):
        raise ValueError('invalid public panel boundary')
    if set(released) != {'RELEASED_40.json', 'AUXILIARY_40.json'}:
        raise ValueError('same-archive public release40 missing')
    release = released['RELEASED_40.json']
    aux_release = released['AUXILIARY_40.json']
    if release['release_time'] != 40 or aux_release['release_time'] != 40:
        raise ValueError('same-archive public release clock mismatch')
    if not np.array_equal(physical, np.asarray(release['physical'], dtype=float)):
        raise ValueError('boundary physical differs from same-archive release40')
    if not np.array_equal(auxiliary, np.asarray(aux_release['auxiliary'], dtype=float)):
        raise ValueError('boundary auxiliary differs from same-archive release40')
    last = history.loc[history.period == 39].sort_values('path_id')
    if len(last) != 32 or not np.array_equal(last.path_id.to_numpy(), np.arange(32)):
        raise ValueError('learner release40 path coverage mismatch')
    if not np.array_equal(last[['next_state', 'next_w', 'action']].to_numpy(float), physical):
        raise ValueError('learner history does not match public release40')
    panel = panel_class(learner, physical.copy(), auxiliary.copy())
    if panel.pending is not None:
        raise ValueError('restored panel has pending issue')
    return panel


def clone_boundary(panel):
    """A fresh independent learner/panel for each prospective replication."""
    if panel.pending is not None or panel.learner.pending is not None:
        raise ValueError('cannot clone an issued transition')
    return copy.deepcopy(panel)
