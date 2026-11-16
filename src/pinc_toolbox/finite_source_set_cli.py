"""JSON transport for the finite four-class retained-set reporter."""
import argparse
import json
from pathlib import Path
from .finite_source_set import finite_source_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object containing reporter keyword arguments')
    result = finite_source_set(**request)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
