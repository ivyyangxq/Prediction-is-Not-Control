"""JSON transport for the experimental shared conditional-rank reporter."""
import argparse
import json
from pathlib import Path
from .joint_component_report import joint_component_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object containing reporter keyword arguments')
    result = joint_component_report(**request)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
