"""Compare a Lean expression-tree planner against the frozen token decoder."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .data import Goal, SPLIT_COUNTS, STRATEGIES
from .experiment import Vocabulary, batches, collate, evaluate, score, write_json
from .model import ModelConfig, ProofModel
from .tree_plan import UnsupportedGoal, build_plan, extract, render_plan
from .verify import check


@torch.no_grad()
def encode_latents(model, examples, batch_size):
    latents = []
    for batch in batches(examples, batch_size):
        _, proofs = collate(batch)
        latents.append(model.encode_proof(proofs).flatten(1))
    return F.normalize(torch.cat(latents), dim=1)


def fit_route_head(latents, seed):
    torch.manual_seed(seed)
    labels = torch.arange(len(latents)).remainder(4).div(2, rounding_mode="floor")
    head = nn.Linear(latents.shape[1], len(STRATEGIES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.02)
    for _ in range(200):
        indices = torch.randint(len(latents), (256,))
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(latents[indices]), labels[indices]).backward()
        optimizer.step()
    with torch.no_grad():
        correct = int(head(latents).argmax(-1).eq(labels).sum())
    return head.eval(), {"correct": correct, "attempts": len(labels)}


def plan_candidates(trees):
    candidates = []
    for tree in trees:
        try:
            plan = build_plan(tree)
        except UnsupportedGoal:
            candidates.append({})
            continue
        proofs = {}
        for route in STRATEGIES:
            try:
                proofs[route] = render_plan(plan, route)
            except UnsupportedGoal:
                continue
        candidates.append(proofs)
    return candidates


def verify_rows(args, goals, rows, rows_per_goal):
    cases = [(goal.statement, row["proof"] or "")
             for goal, group in zip(goals, batches(rows, rows_per_goal), strict=True) for row in group]
    for row, verdict in zip(rows, check(args.lean, args.output, cases, args.verify_workers), strict=True):
        if verdict.status == "checker_error":
            raise RuntimeError(f"Lean verification failed: {verdict.message}")
        row.update(asdict(verdict))
        if row["proof"] is None:
            row["status"] = "unsupported_tree_plan"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(2000,), required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Trusted results directory of the original 2,000-goal run.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lean", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--verify-workers", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("This comparison must run in a Slurm allocation.")
    limit = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 0 < args.threads <= limit or not 0 < args.verify_workers <= limit:
        parser.error("Thread and worker counts must fit the Slurm allocation.")

    start = perf_counter()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    dataset = (args.baseline / "dataset.jsonl").read_bytes()
    goals = [Goal(**json.loads(row)) for row in dataset.splitlines()]
    train = [goal for goal in goals if goal.split == "train"]
    test = [goal for goal in goals if goal.split == "test"]
    expected = SPLIT_COUNTS[args.size]
    if len(train) != expected["train"] or len(test) != expected["test"]:
        parser.error("Baseline dataset does not have the expected training and held-out splits.")
    reference = json.loads((args.baseline / "summary.json").read_text(encoding="utf-8"))
    args.batch_goals = reference["settings"]["batch_goals"]
    args.max_tokens = reference["settings"]["max_tokens"]
    args.overfit_train = False
    baseline = reference["runs"]["strategy"]["evaluation"]["test"]["encoded"]
    checkpoint_bytes = (args.baseline / "strategy.pt").read_bytes()
    checkpoint = torch.load(args.baseline / "strategy.pt", map_location="cpu", weights_only=True)
    vocabulary = Vocabulary(train)
    if vocabulary.items != checkpoint["vocabulary"] or checkpoint["mode"] != "strategy":
        parser.error("Baseline checkpoint and training vocabulary do not match.")
    model = ProofModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state"])
    model.eval()

    def tokenize(split):
        return [(vocabulary.encode(goal.statement), [vocabulary.encode(p) for p in goal.proofs])
                for goal in split]

    test_examples = tokenize(test)
    print(f"Replaying the archived checkpoint on {len(test)} held-out goals", flush=True)
    token_scores, token_rows = evaluate(args, model, vocabulary, test, test_examples, "strategy", "encoded")
    write_json(args.output / "token-decoder.json", token_rows)
    original = json.loads((args.baseline / "strategy-test-encoded.json").read_text(encoding="utf-8"))
    same_outputs = sum((a["goal_id"], a["proof"], a["status"]) == (b["goal_id"], b["proof"], b["status"])
                       for a, b in zip(token_rows, original, strict=True))
    print(f"Token baseline accepted {token_scores['valid']}/{len(token_rows)} proofs; "
          f"{same_outputs} outputs matched the archived run", flush=True)

    route_start = perf_counter()
    latents = encode_latents(model, tokenize(train), args.batch_goals)
    head, head_fit = fit_route_head(latents, reference["settings"]["seed"])
    torch.save(head.state_dict(), args.output / "route-head.pt")
    fit_s = perf_counter() - route_start
    route_start = perf_counter()
    with torch.no_grad():
        routes = head(encode_latents(model, test_examples, args.batch_goals)).argmax(-1).tolist()
    route_s = perf_counter() - route_start
    print(f"Learned route head matched {head_fit['correct']}/{head_fit['attempts']} training labels", flush=True)

    extract_start = perf_counter()
    trees = extract(args.lean, args.output, [goal.statement for goal in test], args.verify_workers)
    extract_s = perf_counter() - extract_start
    write_json(args.output / "lean-trees.json", [{"goal_id": g.id, **t} for g, t in zip(test, trees, strict=True)])
    plan_start = perf_counter()
    candidates = plan_candidates(trees)
    plan_s = perf_counter() - plan_start
    rows = []
    for goal, proofs, choices in zip(test, candidates, batches(routes, 4), strict=True):
        for index, choice in enumerate(choices):
            route = STRATEGIES[choice]
            rows.append({"goal_id": goal.id, "expected": STRATEGIES[index // 2], "selected": route,
                         "rendering": index % 2, "proof": proofs.get(route)})
    verify_start = perf_counter()
    verify_rows(args, test, rows, 4)
    verify_s = perf_counter() - verify_start
    scores = score(rows)
    scores["correct_routes"] = sum(row["selected"] == row["expected"] for row in rows)
    write_json(args.output / "tree-decoder.json", rows)

    # Measure how much the fixed rules solve by enumerating both routes without latents.
    symbolic = [{"goal_id": g.id, "requested": route, "proof": proofs.get(route)}
                for g, proofs in zip(test, candidates, strict=True) for route in STRATEGIES]
    verify_rows(args, test, symbolic, 2)
    write_json(args.output / "symbolic-enumeration.json", symbolic)
    symbolic_scores = {"valid": sum(row["status"] == "valid" for row in symbolic),
                       "attempts": len(symbolic), "goals": len(test),
                       "paired_correct": sum(all(row["status"] == "valid" and row["strategy"] == row["requested"]
                                                 for row in pair) for pair in batches(symbolic, 2))}
    summary = {
        "slurm_job": os.environ["SLURM_JOB_ID"], "torch": torch.__version__,
        "baseline_run": reference["slurm_job"], "baseline_step": checkpoint["step"],
        "dataset_sha256": sha256(dataset).hexdigest(), "checkpoint_sha256": sha256(checkpoint_bytes).hexdigest(),
        "identical_baseline_outputs": same_outputs, "archived_token_scores": baseline, "token_decoder": token_scores,
        "tree_decoder": scores, "route_head_train": head_fit, "symbolic_enumeration": symbolic_scores,
        "seconds": {"fit_route_head_including_encoding": fit_s, "test_route_including_encoding": route_s,
                    "lean_extract": extract_s, "build_plans": plan_s, "verify_structured": verify_s,
                    "total": perf_counter() - start},
    }
    write_json(args.output / "summary.json", summary)
    print(f"Tree decoder: {scores}\nSymbolic enumeration: {symbolic_scores}", flush=True)


if __name__ == "__main__":
    main()
