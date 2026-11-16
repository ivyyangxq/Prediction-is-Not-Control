"""Candidate future conditioning using MAIN and acquired probe observations."""
from pathlib import Path
import json,lzma,pickle,sys
import numpy as np
HERE=Path(__file__).resolve().parent
COND=HERE.parent.parent/'track2_errors/paired_monitor_conditioning_attempt001'
sys.path.insert(0,str(COND))
from observed_pairs import assemble_pairs
from public_posterior_with_pairs import filter_public_history

def expanded_rows(archive):
    found={}
    with lzma.open(archive,'rb') as stream:
        while True:
            row=pickle.load(stream)
            if row[0]=='end':break
            if row[0]=='raw' and row[1] in ('MAIN_OBSERVATIONS.json','PAIRED_PROBE_ENDPOINTS.json'):
                found[row[1]]=json.loads(row[2])
    if len(found)!=2:raise ValueError('expanded observed transcript incomplete')
    return assemble_pairs(found['MAIN_OBSERVATIONS.json'],found['PAIRED_PROBE_ENDPOINTS.json'])

def science_class(base):
    class ExpandedScience(base.Science):
        def condition(self,selected_index):
            if self._observed_private is None:raise RuntimeError('observed boundary absent')
            directory=self.out/'selected_public';directory.mkdir()
            records,history,readback=self.extractor.read_public_until40(self.selected_archive)
            arrays,table,learner_history,issued,boundary,checks=self.extractor.assemble(records,history)
            pairs,refusals=expanded_rows(self.selected_archive)
            boundary.update(H=True,H_status='scalar_serial_paired_union',extra_observation_dates=len(pairs))
            self.write_npz(directory/'FILTER_INPUT.npz',arrays)
            self.write_json(directory/'EXTRA_OBSERVATIONS.json',dict(pairs=pairs,refusals=refusals))
            self.write(directory/'PUBLIC_TRANSITIONS.csv',table.to_csv(index=False,float_format='%.17g'))
            self.write(directory/'PUBLIC_LEARNER_HISTORY.csv',learner_history.to_csv(index=False,float_format='%.17g'))
            self.write_json(directory/'PUBLIC_MONITOR_ISSUED.json',issued)
            self.write_json(directory/'PUBLIC_BOUNDARY.json',boundary)
            self.write_json(directory/'PROVENANCE_NOT_FILTER_INPUT.json',dict(selected_index=selected_index,
                archive=str(self.selected_archive),selection='fixed prospective union alarm; no source-future selection',**readback,**checks))
            panel=self.restore.restore_boundary(self.selected_archive,directory/'PUBLIC_BOUNDARY.json',self.b.front.Panel)
            if type(panel.learner)._q is not self.b.BellmanLearner._q:raise RuntimeError('Bellman learner lost')
            posteriors={};saved={}
            for candidate in base.CELLS:
                posterior=filter_public_history(candidate=candidate,paired_observations=pairs,**arrays)
                self.filter_calls+=1
                if posterior.final_date!=40:raise ValueError('incorrect expanded posterior date')
                posteriors[candidate]=(posterior.means.copy(),posterior.covariances.copy())
                saved[candidate+'_means']=posterior.means;saved[candidate+'_covariances']=posterior.covariances
            self.write_npz(directory/'POSTERIORS.npz',saved)
            return dict(panel=panel,posteriors=posteriors)
    return ExpandedScience
