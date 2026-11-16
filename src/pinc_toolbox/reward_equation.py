"""Evaluate a prescribed scalar reward with explicitly declared inputs."""
from __future__ import annotations

import ast
import keyword
from math import isfinite

import numpy as np


_FUNCTIONS = {"abs": np.abs, "exp": np.exp, "log": np.log, "sqrt": np.sqrt}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)


class RewardEquation:
    """A fixed arithmetic expression, with no access to the surrounding data."""

    def __init__(self, expression, variables, parameters):
        self.variables = tuple(variables)
        self.parameters = tuple((name, float(value)) for name, value in parameters.items())
        names = self.variables + tuple(name for name, _ in self.parameters)
        if (len(set(names)) != len(names) or any(
                not isinstance(name, str) or not name.isidentifier()
                or keyword.iskeyword(name) or name in _FUNCTIONS for name in names)):
            raise ValueError("reward variables and parameters need distinct valid identifiers")
        if any(not isfinite(value) for _, value in self.parameters):
            raise ValueError("reward parameters must be finite")
        tree = ast.parse(expression, mode="eval")
        calls = {node.func for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
                 and len(node.args) == 1 and not node.keywords}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and (node.id in names or node in calls):
                continue
            if isinstance(node, ast.Constant) and type(node.value) in (int, float) and isfinite(node.value):
                node.value = float(node.value)
                continue
            if isinstance(node, ast.Call) and node.func in calls and len(node.args) == 1 and not node.keywords:
                continue
            if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, *_OPERATORS)):
                continue
            raise ValueError(f"reward equation contains an undeclared name or unsupported expression: {ast.unparse(node)}")
        self.expression = expression
        self._code = compile(tree, "<prescribed reward>", "eval")

    def __call__(self, inputs):
        names = {name: np.asarray(inputs[name], dtype=float) for name in self.variables}
        with np.errstate(all="ignore"):
            return eval(self._code,
                        {"__builtins__": {}, **_FUNCTIONS}, {**dict(self.parameters), **names})
