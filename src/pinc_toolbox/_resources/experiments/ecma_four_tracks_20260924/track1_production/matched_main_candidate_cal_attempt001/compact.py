"""Incremental, lossless recording of actual learner objects; no scientific calls."""
from pathlib import Path
import pickle, cloudpickle, lzma, copy, shutil
import pandas as pd

RESERVE=512*1024**2
ALLOWANCE=int(1.3*1024**3)

class ResourceBudget:
    def __init__(self,root,allowance=ALLOWANCE,reserve=RESERVE):
        self.root=Path(root);self.allowance=allowance;self.reserve=reserve
        self.reporting=False
    def check(self,extra=0):
        used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        limit=self.allowance if self.reporting else max(0,self.allowance-16*1024**2)
        if used+extra>limit:raise OSError('new-output allowance reached; preserve 16MiB for final accounting')
        if shutil.disk_usage(self.root).free-extra<self.reserve:raise OSError('free-space reserve reached')
        return used

class EpisodeWriter:
    def __init__(self,path,budget):
        self.path=Path(path);self.budget=budget;budget.check()
        self.file=self.path.open('xb');self.compressor=lzma.LZMACompressor(preset=6)
        self.previous={};self.names=[];self.closed=False;self.healthy=True
    def _write(self,record):
        if not self.healthy:raise OSError('archive writer has failed')
        try:
            self.budget.check()
            data=self.compressor.compress(pickle.dumps(record,protocol=5))
            self.budget.check(len(data));self.file.write(data);self.file.flush()
            self.budget.check()
        except BaseException:
            self.healthy=False;raise
    def _name(self,name):
        if name in self.names:raise ValueError('duplicate record name')
        self.names.append(name)
    def raw(self,name,data):
        self._name(name);self._write(('raw',name,data))
    def append_raw(self,name,data):
        if name not in self.names:self._name(name)
        self._write(('append',name,data))
    def snapshot(self,name,learner):
        self._name(name);chain=str(Path(name).parent);hist=learner.history
        assert isinstance(hist.index,pd.RangeIndex) and hist.index.start==0 and hist.index.step==1
        base=chain if chain in self.previous else ('source' if chain.startswith('source/') and 'source' in self.previous else None)
        prev=None if base is None else self.previous[base]
        if prev is not None and (len(hist)<len(prev) or not hist.iloc[:len(prev)].equals(prev)):base=None;prev=None
        delta=hist.copy() if prev is None else hist.iloc[len(prev):].copy()
        # Shallow shell leaves the active learner untouched, even on write failure.
        shell=copy.copy(learner);shell.history=None
        self._write(('snapshot',name,chain,base,delta,cloudpickle.dumps(shell,protocol=5)))
        self.previous[chain]=hist.copy()
    def finish(self):
        if self.closed:return
        try:
            self._write(('end',self.names))
            data=self.compressor.flush();self.budget.check(len(data))
            self.file.write(data);self.file.flush();self.budget.check()
        finally:self.file.close();self.closed=True
    def abort(self):
        """Keep bytes already written. No terminal marker and no complete claim."""
        if self.closed:return
        try:
            if self.healthy:
                data=self.compressor.flush();self.budget.check(len(data));self.file.write(data);self.file.flush()
        finally:self.file.close();self.closed=True

def records(path,*,allow_incomplete=False):
    """Default reader rejects a truncated prefix, including a whole-record boundary.

    Diagnostic allow_incomplete yields recoverable records only. It never supplies
    a completion receipt; production scoring must use the default strict reader.
    """
    previous={};names=[];appended={}
    with lzma.open(path,'rb') as f:
        while True:
            try:r=pickle.load(f)
            except (EOFError,lzma.LZMAError):
                if allow_incomplete:
                    for name,data in appended.items():yield name,data
                    return
                raise ValueError('incomplete episode archive')
            if r[0]=='end':
                if names!=r[1]:raise ValueError('terminal record list mismatch')
                if f.read(1):raise ValueError('unexpected trailing archive data')
                for name,data in appended.items():yield name,data
                return
            if r[0]=='append':
                if r[1] not in appended:
                    if r[1] in names:raise ValueError('append collides with another record')
                    names.append(r[1]);appended[r[1]]=b''
                appended[r[1]]+=r[2];continue
            if r[1] in names:raise ValueError('duplicate archived record')
            names.append(r[1])
            if r[0]=='raw':yield r[1],r[2];continue
            if r[0]!='snapshot':raise ValueError('unknown archive record')
            _,name,chain,base,delta,payload=r
            hist=delta.copy() if base is None else pd.concat([previous[base],delta],ignore_index=True)
            learner=cloudpickle.loads(payload);learner.history=hist
            previous[chain]=hist
            yield name,learner
