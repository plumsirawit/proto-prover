"""Compile supported Lean inequality trees into explicit checked-proof plans."""

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


class UnsupportedGoal(ValueError):
    """The extracted expression is outside the planner's rule set."""


def extract_batch(lean, scratch, statements):
    extractor = Path(__file__).with_name("goal_tree.lean").read_text(encoding="utf-8")
    blocks = [extractor]
    for index, statement in enumerate(statements):
        # These temporary axioms provide types for the inspector. The verifier
        # uses independent Lean processes and never imports this file.
        blocks.append(f"axiom goal_{index} {statement}\ninspect_goal_tree goal_{index}")
    with TemporaryDirectory(dir=scratch, prefix="goal-tree-") as directory:
        path = Path(directory, "Goals.lean")
        path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        result = subprocess.run([lean, "-j", "1", str(path)], cwd=directory, capture_output=True,
                                text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"Lean expression extraction failed:\n{result.stdout}\n{result.stderr}")
    trees = {}
    for line in result.stdout.splitlines():
        if line.startswith("GOAL_TREE "):
            _, name, payload = line.split(" ", 2)
            trees[name] = json.loads(payload)
    if len(trees) != len(statements):
        raise RuntimeError(f"Lean emitted {len(trees)} trees for {len(statements)} goals")
    return [trees[f"goal_{index}"] for index in range(len(statements))]


def extract(lean, scratch, statements, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(extract_batch, lean, scratch, statements[i:i + 24])
                 for i in range(0, len(statements), 24)]
        return [tree for task in tasks for tree in task.result()]


def freeze(tree):
    return tuple(freeze(child) if isinstance(child, list) else child for child in tree)


@dataclass(frozen=True)
class Addition:
    left: "Plan"
    right: "Plan"


@dataclass(frozen=True)
class Reflexivity:
    expression: tuple


@dataclass(frozen=True)
class HypothesisPath:
    left: tuple
    right: tuple
    names: tuple[str, ...]


Plan = Addition | Reflexivity | HypothesisPath


def find_path(graph, left, right):
    queue = deque([(left, ())])
    visited = {left}
    while queue:
        node, path = queue.popleft()
        if node == right:
            return path
        for target, name in graph[node]:
            if target not in visited:
                queue.append((target, (*path, name)))
                visited.add(target)
    raise UnsupportedGoal(f"No inequality hypotheses connect {left} to {right}")


def build_plan(tree):
    graph = defaultdict(list)
    for name, left, right in tree["assumptions"]:
        graph[freeze(left)].append((freeze(right), name))

    def plan(left, right):
        if left == right:
            return Reflexivity(left)
        if left[0] == right[0] == "add":
            return Addition(plan(left[1], right[1]), plan(left[2], right[2]))
        return HypothesisPath(left, right, find_path(graph, left, right))

    return plan(freeze(tree["left"]), freeze(tree["right"]))


def chain_term(names):
    term = names[-1]
    for name in reversed(names[:-1]):
        term = f"(Nat.le_trans {name} {term})"
    return term


def monotonicity_term(plan):
    if isinstance(plan, Addition):
        return f"(Nat.add_le_add {monotonicity_term(plan.left)} {monotonicity_term(plan.right)})"
    if isinstance(plan, Reflexivity):
        return "(Nat.le_refl _)"
    return chain_term(plan.names)


def leaf_bounds(plan):
    if isinstance(plan, Addition):
        yield from leaf_bounds(plan.left)
        yield from leaf_bounds(plan.right)
    elif isinstance(plan, HypothesisPath):
        yield plan


def render_plan(plan, strategy):
    if strategy == "monotonicity":
        return f"exact {monotonicity_term(plan)}"
    if strategy != "gap_witness":
        raise ValueError(f"Unknown strategy {strategy}")
    bounds = tuple(dict.fromkeys(leaf_bounds(plan)))
    if not bounds:
        raise UnsupportedGoal("A witness proof needs at least one unequal leaf pair")
    lines = [f"obtain ⟨gap{i}, eqn{i}⟩ := Nat.exists_eq_add_of_le {chain_term(bound.names)}"
             for i, bound in enumerate(bounds)]
    equations = ", ".join(f"eqn{i}" for i in range(len(bounds)))
    return "\n".join([*lines, f"rw [{equations}]", "omega"])
