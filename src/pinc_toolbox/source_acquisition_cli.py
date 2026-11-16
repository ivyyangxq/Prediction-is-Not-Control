"""JSON bridge for a qualified native-query snapshot and acquisition decision."""
import argparse,json
from pathlib import Path
from .source_acquisition import recommend_source_action


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    if a.input.resolve()==a.output.resolve():p.error('input and output must differ')
    result=recommend_source_action(**json.loads(a.input.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
