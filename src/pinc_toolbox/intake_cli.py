"""Interactive data import and guided GQ execution."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .model_adapter import DetectorConfig, EnvironmentSpec, ReleasedData, validate_environment


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(prompt + suffix + ": ").strip()
    return value if value else (default or "")


def _csv_names(value: str) -> tuple[str, ...]:
    names = tuple(x.strip() for x in value.split(",") if x.strip())
    if not names:
        raise ValueError("at least one column is required")
    return names


def interactive_request() -> dict[str, Any]:
    state = _csv_names(_ask("Public state columns (comma separated)"))
    action_raw = _ask("Executed action column; type none for actionless G-only", "action")
    action = None if action_raw.lower() in {"none", "null", "-"} else action_raw
    reward = _ask("Observed reward column", "reward")
    next_state = _csv_names(_ask("Next-state columns (comma separated)"))
    release = _ask("Outcome release-date column", "release_date")
    decision = _ask("Decision-date column", "decision_date")
    hidden_raw = _ask("Hidden columns (comma separated; blank if none)", "")
    hidden = tuple(x.strip() for x in hidden_raw.split(",") if x.strip())
    q_applicable = _ask("Is an economic action/Q branch applicable? (y/n)", "y").lower() in {"y", "yes"}
    grid_raw = _ask("Action grid (comma separated; blank for G-only)", "0,0.09,0.18")
    grid = () if not grid_raw.strip() else tuple(float(x.strip()) for x in grid_raw.split(","))
    eta = _ask("Eta", "2")
    fit_window = int(_ask("G fit window", "64"))
    monitoring_window = int(_ask("Monitoring window", "8"))
    q_mode = _ask("Q mode (none/finite/infinite)", "finite" if q_applicable else "none").lower()
    q_horizon = None if q_mode != "finite" else int(_ask("Finite Q horizon", "10"))
    table_path = Path(_ask("Released data JSON path"))
    table = json.loads(table_path.read_text())
    spec = EnvironmentSpec(state=state, action=action, reward=reward, next_state=next_state,
                           release_date=release, decision_date=decision, hidden=hidden,
                           q_applicable=q_applicable, action_is_structural=q_applicable)
    data = ReleasedData(table=table, action_grid=grid)
    config = DetectorConfig(eta=eta, fit_window=fit_window, monitoring_window=monitoring_window,
                            q_mode=q_mode, q_horizon=q_horizon)
    report = validate_environment(spec, data, config)
    report["data_json"] = str(table_path)
    report["perceived_model"] = "provided_to_host_runner"
    return report


def _choose(prompt, options, *, multiple=False):
    options = list(options)
    if not options:
        raise ValueError(f"no options available for {prompt}")
    print(prompt)
    for index, value in enumerate(options, 1):
        print(f"  {index}. {value}")
    while True:
        answer = _ask("Select numbers, comma separated or all" if multiple else "Select a number", "all" if multiple else "1")
        try:
            selected = (list(range(1, len(options) + 1)) if multiple and answer.lower() == "all"
                        else [int(item) for item in answer.split(",")])
        except ValueError:
            selected = []
        if (selected and all(1 <= index <= len(options) for index in selected)
                and len(set(selected)) == len(selected) and (multiple or len(selected) == 1)):
            values = [options[index - 1] for index in selected]
            return values if multiple else values[0]
        print("Choose from the numbered options.")


def _read_table(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, float_precision="round_trip")
    if path.suffix.lower() == ".json":
        return pd.read_json(path, precise_float=True, convert_dates=False)
    raise ValueError("guided data must be a CSV or JSON file")


def guided_run(table, *, macro_path=None):
    from .guided_gq import MODES, panel_history, run_panel
    from .reward_equation import RewardEquation

    if macro_path is None and _choose("Source of W", ["Panel columns", "Separate macro file"]) == "Separate macro file":
        macro_path = Path(_ask("Macro CSV or JSON data path"))
    macro = None
    macro_label = None
    macro_config = None
    if macro_path is not None:
        macro_table = _read_table(macro_path)
        value_column = _choose("Macro W value column", [name for name in macro_table
                                                       if pd.api.types.is_numeric_dtype(macro_table[name])])
        published = _choose("Macro publication timestamp column", [name for name in macro_table
                                                                    if name != value_column])
        macro = (macro_table, value_column, published)
        macro_label = f"{value_column} (macro W)"
        macro_config = {"file": str(macro_path), "value_column": value_column,
                        "publication_column": published}

    numeric = [name for name in table if pd.api.types.is_numeric_dtype(table[name])]
    columns = {}
    for field, label in (("state", "First current state"), ("w", "Second current state"),
                         ("action", "Executed choice"), ("next_state", "Next first state"),
                         ("next_w", "Next second state"), ("period", "Decision date"),
                         ("release_period", "Transition release date")):
        if macro is not None and field in ("w", "next_w"):
            continue
        pool = table.columns if field in ("period", "release_period") else numeric
        columns[field] = _choose(label, [name for name in pool if name not in columns.values()])
    remaining = [name for name in numeric if name not in columns.values()]
    path_column = _choose("Panel identifier (None for a single path)", [None, *remaining])
    if path_column is not None:
        columns["path_id"] = path_column
    history = panel_history(table, columns, macro=macro)
    mode = _choose("Learning mode and source of reward inputs", MODES)
    exploration = 0.0
    if mode != MODES[1]:
        exploration = _choose("Probability of exploring a supported candidate action",
                              [0.0, 0.05, 0.10, 0.20])
    path = _choose("Path to query", sorted(history.path_id.unique().tolist()))
    dates = sorted(history.loc[history.path_id.eq(path) & history.previous_action.notna(), "period"].tolist())
    first_release = history.release_period.min()
    calendar = history.attrs.get("decision_calendar")
    labels = {date if calendar is None else calendar[date]: date for date in dates if date >= first_release}
    start = labels[_choose("First decision date", labels)]
    end = labels[_choose("Last decision date", [label for label in reversed(labels) if labels[label] >= start])]
    prefix = history.loc[(history.period < start) & (history.release_period <= start)]
    grid = sorted(prefix.loc[prefix.action.notna(), "action"].unique().tolist())
    candidates = _choose("Candidate action values observed in the released prefix", grid, multiple=True)
    max_window = int(start - prefix.period.min())
    window = _choose("G fit window in periods", list(dict.fromkeys([min(max_window, 24),
                     *[size for size in (8, 16, 32, 64) if size <= max_window], max_window])))
    horizon = _choose("Planning horizon", [10, 1, 5, 20, "Infinite"])
    discount = _choose("Discount factor", [.95, .9, .98, .99, 0.])
    input_labels = {columns["state"]: "state", macro_label or columns["w"]: "w", columns["action"]: "action"}
    label = ("Observed variables entering the reward equation" if mode == MODES[1]
             else "Predicted state and selected-action variables entering the reward equation")
    selected = _choose(label, input_labels, multiple=True)
    inputs = {}
    for i, name in enumerate(selected, 1):
        # Fixed aliases also work for imported headers with spaces or punctuation.
        alias = f"v{i}"
        inputs[alias] = input_labels[name]
        print(f"  {alias} = {name}")
    while True:
        expression = _ask("Reward equation, using the aliases above and named numerical parameters")
        try:
            tree = ast.parse(expression, mode="eval")
            function_nodes = {node.func for node in ast.walk(tree) if isinstance(node, ast.Call)}
            params = sorted({node.id for node in ast.walk(tree)
                             if isinstance(node, ast.Name) and node not in function_nodes} - set(inputs))
            if params and _choose("Treat " + ", ".join(params) + " as numerical parameters?",
                                  ["Yes", "Edit equation"]) == "Edit equation":
                continue
            values = {name: float(_ask(f"Value of {name}")) for name in params}
            RewardEquation(expression, inputs, values)
            break
        except (SyntaxError, ValueError, OverflowError) as error:
            print(f"Invalid reward equation: {error}")
    return run_panel(history, {"columns": columns, "path": path, "start": start, "end": end,
                               "window": window, "mode": mode, "grid": grid, "candidates": candidates,
                               "horizon": None if horizon == "Infinite" else horizon, "discount": discount,
                               "exploration_probability": exploration, "exploration_seed": 0,
                               "macro": macro_config,
                               "reward": {"expression": expression, "inputs": inputs, "parameters": values}})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PINC environment intake")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--interactive", action="store_true", help="collect a legacy monitoring intake report")
    modes.add_argument("--guided", action="store_true", help="import data, choose columns and run GQ")
    parser.add_argument("--data", type=Path, help="CSV or JSON data for the guided run")
    parser.add_argument("--macro", type=Path, help="optional separate CSV or JSON file for observable W")
    parser.add_argument("--output", type=Path, help="write the result JSON")
    args = parser.parse_args(argv)
    if args.guided:
        path = args.data or Path(_ask("CSV or JSON data path"))
        try:
            report = guided_run(_read_table(path), macro_path=args.macro)
        except (OSError, ValueError, ArithmeticError, EOFError) as error:
            parser.error(str(error))
    else:
        report = interactive_request()
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=not args.guided) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
