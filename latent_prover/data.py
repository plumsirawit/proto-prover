"""Original paired proofs for a narrow experiment on natural-number order."""

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations, product
import json
from pathlib import Path
import random


STRATEGIES = ("monotonicity", "gap_witness")

# Every expression tree denotes the same operation on its two sides. Integer
# leaves index inequality hypotheses. The string leaf is a shared offset.
TRAIN = {
    "pair_shift_right": ((0, 1), "k"),
    "pair_shift_left": ("k", (0, 1)),
    "left_branch_shift": ((0, "k"), 1),
    "right_branch_shift": (0, (1, "k")),
    "balanced_triple": ((0, 1), (2, "k")),
    "right_spine": (0, (1, (2, "k"))),
    "left_spine": (((0, "k"), 1), 2),
    "inner_right_shift": ((0, (1, "k")), 2),
}
DEV = {
    "right_inner_left_shift": (0, ("k", 1)),
    "triple_inner_left_shift": ((0, ("k", 1)), 2),
}
TEST = {
    "left_inner_left_shift": (("k", 0), 1),
    "right_pair_then_shift": (0, ((1, 2), "k")),
}
SPLIT_COUNTS = {
    24: {"train": 16, "dev": 2, "test": 4, "seen": 2},
    200: {"train": 128, "dev": 24, "test": 24, "seen": 24},
    2000: {"train": 1280, "dev": 240, "test": 240, "seen": 240},
}


@dataclass(frozen=True)
class Goal:
    id: str
    family: str
    split: str
    statement: str
    # Fixed order: monotonicity renderings 0 and 1, then gap renderings 0 and 1.
    proofs: tuple[str, str, str, str]


def indices(tree):
    if isinstance(tree, tuple):
        return indices(tree[0]) + indices(tree[1])
    return [tree] if isinstance(tree, int) else []


def expression(tree, side, offset):
    if isinstance(tree, tuple):
        return f"({expression(tree[0], side, offset)} + {expression(tree[1], side, offset)})"
    return f"{side}{tree}" if isinstance(tree, int) else str(offset)


def monotonicity_term(tree, hypotheses):
    if isinstance(tree, tuple):
        left, right = (monotonicity_term(child, hypotheses) for child in tree)
        return f"(Nat.add_le_add {left} {right})"
    return hypotheses[tree] if isinstance(tree, int) else "(Nat.le_refl _)"


def relabel_tree(tree, order):
    if isinstance(tree, tuple):
        return tuple(relabel_tree(child, order) for child in tree)
    return order[tree] if isinstance(tree, int) else tree


@lru_cache
def variants(family, arity):
    leaf_orders = permutations(range(arity))
    chains = tuple(product((1, 2, 3), repeat=arity))
    choices = list(product(leaf_orders, chains, (False, True)))
    random.Random(family).shuffle(choices)
    return choices


def make_goal(family, tree, split, depths, offset, reverse_assumptions):
    leaves = sorted(depths)
    variables = [f"{side}{i}" for i in leaves for side in ("a", "b")]
    variables.extend(f"m{i}{suffix}" for i in leaves
                     for suffix in ("", "_2")[:depths[i] - 1])
    assumptions = []
    term = {}
    for i in leaves:
        if depths[i] == 3:
            assumptions.extend((f"(h{i} : a{i} ≤ m{i})", f"(j{i} : m{i} ≤ m{i}_2)",
                                f"(j{i}_2 : m{i}_2 ≤ b{i})"))
            term[i] = f"(Nat.le_trans h{i} (Nat.le_trans j{i} j{i}_2))"
        elif depths[i] == 2:
            assumptions.extend((f"(h{i} : a{i} ≤ m{i})", f"(j{i} : m{i} ≤ b{i})"))
            term[i] = f"(Nat.le_trans h{i} j{i})"
        else:
            assumptions.append(f"(h{i} : a{i} ≤ b{i})")
            term[i] = f"h{i}"
    if reverse_assumptions:
        assumptions.reverse()
    statement = (f"({' '.join(variables)} : Nat) {' '.join(assumptions)} : "
                 f"{expression(tree, 'a', offset)} ≤ {expression(tree, 'b', offset)}")

    inline = f"exact {monotonicity_term(tree, term)}"
    named = []
    names = {}
    for i in leaves:
        names[i] = f"bound{i}"
        named.append(f"have bound{i} : a{i} ≤ b{i} := {term[i]}")
    named.append(f"exact {monotonicity_term(tree, names)}")

    witnesses = []
    for tactic, gap, eq in [("obtain", "gap", "eqn"), ("rcases", "delta", "step")]:
        lines = []
        for i in leaves:
            proof = f"Nat.exists_eq_add_of_le {term[i]}"
            pattern = f"⟨{gap}{i}, {eq}{i}⟩"
            lines.append(f"obtain {pattern} := {proof}" if tactic == "obtain"
                         else f"rcases {proof} with {pattern}")
        if tactic == "obtain":
            lines.append(f"rw [{', '.join(f'{eq}{i}' for i in leaves)}]")
        else:
            lines.extend(f"rw [{eq}{i}]" for i in reversed(leaves))
        lines.append("omega")
        witnesses.append("\n".join(lines))

    return Goal(f"{family}-{offset:04}", family, split, statement,
                (inline, "\n".join(named), *witnesses))


def generate(size):
    shapes = {"train": TRAIN, "dev": DEV, "test": TEST, "seen": TRAIN}
    goals = []
    for split, count in SPLIT_COUNTS[size].items():
        families = tuple(shapes[split].items())
        for index in range(count):
            family, tree = families[index % len(families)]
            within_family, family_index = divmod(index, len(families))
            arity = len(indices(tree))
            # Keep the smallest pilot short and preserve it inside larger runs.
            if within_family < 2:
                pattern = within_family + family_index
                depths = [1 + ((pattern >> i) & 1) for i in range(arity)]
                reverse_assumptions = False
            else:
                choices = variants(family, arity)
                option = within_family - 2
                if split == "seen":
                    option = len(choices) - option - 1
                order, depths, reverse_assumptions = choices[option % len(choices)]
                tree = relabel_tree(tree, order)
            partition = 4096 if split == "seen" else 0
            goals.append(make_goal(family, tree, split, dict(enumerate(depths)),
                                   partition + index + 1, reverse_assumptions))
    assert len({goal.id for goal in goals}) == len(goals)
    assert len({goal.statement for goal in goals}) == len(goals)
    return goals


def write_dataset(path: Path, goals):
    with path.open("w", encoding="utf-8") as stream:
        for goal in goals:
            stream.write(json.dumps(asdict(goal), ensure_ascii=False) + "\n")
    return dict(Counter(goal.split for goal in goals))
