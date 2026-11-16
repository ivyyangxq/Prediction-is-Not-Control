"""Full rescan on every check, with one target stat per entry.

No ledger, filesystem notification dependency, skipped checks or persistent
metadata cache. Match the deployed Python3.9 Path.is_file ignored-error rules.
Like the original walk this is a live traversal, not an atomic filesystem view.
"""
from pathlib import Path
import os
import stat
import shutil

RESERVE=512*1024**2
ALLOWANCE=int(1.3*1024**3)
IGNORED_ERRNOS=(2,20,9,62)  # deployed macOS Python3.9 pathlib._IGNORED_ERROS
IGNORED_WINERRORS=(21,123,1921)

def ignored_error(exc):
    return (getattr(exc,'errno',None) in IGNORED_ERRNOS or
            getattr(exc,'winerror',None) in IGNORED_WINERRORS)

def tree_bytes(root):
    root=Path(root)
    if not root.is_dir():return 0
    pending=[str(root)];total=0
    while pending:
        directory=pending.pop()
        try:
            with os.scandir(directory) as iterator:entries=list(iterator)
        except PermissionError:
            # Path.rglob's directory selectors likewise skip unreadable listings.
            continue
        for entry in entries:
            try:
                info=entry.stat(follow_symlinks=True)
            except OSError as exc:
                if ignored_error(exc):continue
                raise
            except ValueError:
                continue
            if stat.S_ISREG(info.st_mode):total+=info.st_size
            elif stat.S_ISDIR(info.st_mode) and not entry.is_symlink():
                pending.append(entry.path)
    return total

class ResourceBudget:
    def __init__(self,root,allowance=ALLOWANCE,reserve=RESERVE):
        self.root=Path(root);self.allowance=allowance;self.reserve=reserve
        self.reporting=False
    def check(self,extra=0):
        used=tree_bytes(self.root)
        limit=self.allowance if self.reporting else max(0,self.allowance-16*1024**2)
        if used+extra>limit:
            raise OSError('new-output allowance reached; preserve 16MiB for final accounting')
        if shutil.disk_usage(self.root).free-extra<self.reserve:
            raise OSError('free-space reserve reached')
        return used
