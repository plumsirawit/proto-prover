"""Train, verify and compare models on the paired Lean proof experiment."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import SPLIT_COUNTS, STRATEGIES, generate, write_dataset
from .model import ModelConfig, ProofModel, strategy_loss
from .verify import TOKEN, check


PAD, BOS, EOS, UNK, NL = range(5)
MODES = ("reconstruction", "strategy", "label", "canonical")
DEFAULT_MODES = ("reconstruction", "strategy", "label")


def tokens(text):
    return [token for line in text.split("\n") for token in [*TOKEN.findall(line), "<NL>"]][:-1]


class Vocabulary:
    def __init__(self, goals):
        proof_tokens = {token for goal in goals for proof in goal.proofs for token in tokens(proof)}
        goal_tokens = {token for goal in goals for token in tokens(goal.statement)}
        alphabet = (proof_tokens | goal_tokens | set("0123456789")) - {"<NL>"}
        self.items = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<NL>", *sorted(alphabet)]
        self.ids = {key: index for index, key in enumerate(self.items)}
        self.forbidden = [index for index, key in enumerate(self.items)
                          if index not in (EOS, NL) and key not in proof_tokens]

    def encode(self, text):
        return torch.tensor([BOS, *(self.ids.get(t, UNK) for t in tokens(text)), EOS])

    def decode(self, ids):
        if EOS not in ids:
            return None
        return " ".join(self.items[index] for index in ids[:ids.index(EOS)]).replace(" <NL> ", "\n")


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def collate(examples):
    goal_ids, proof_ids = [], []
    for goal, proofs in examples:
        goal_ids.extend([goal] * 4)
        proof_ids.extend(proofs)
    return pad_sequence(goal_ids, batch_first=True), pad_sequence(proof_ids, batch_first=True)


def objective(model, goal, proofs, mode, weight):
    rows = torch.arange(proofs.shape[0])
    if mode == "label":
        latent = model.encode_label(len(rows), goal.device)
    else:
        latent = model.encode_proof(proofs)
    # Decode either rendering to a canonical proof for its strategy. In the
    # reconstruction control, train on exact input text instead.
    target = proofs if mode == "reconstruction" else proofs[rows - rows.remainder(2)]
    memory, pad = model.memory(goal, latent)
    prediction = model.decode(target[:, :-1], memory, pad)
    reconstruction = F.cross_entropy(prediction.flatten(0, 1), target[:, 1:].flatten(), ignore_index=PAD)
    loss = reconstruction + weight * strategy_loss(latent) if mode == "strategy" else reconstruction
    return loss, reconstruction


def train(args, mode, config, train_rows, dev_rows, directory, vocabulary):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ProofModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best, best_step = math.inf, 0
    order = list(train_rows)
    step = 0
    start = perf_counter()
    while step < args.steps:
        random.shuffle(order)
        for batch in batches(order, args.batch_goals):
            model.train()
            goal, proofs = collate(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, reconstruction = objective(model, goal, proofs, mode, args.contrastive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step in (1, args.steps) or step % args.validate_every == 0:
                model.eval()
                with torch.no_grad():
                    dev_losses = []
                    for dev_batch in batches(dev_rows, args.batch_goals):
                        _, dev_loss = objective(model, *collate(dev_batch), mode, args.contrastive_weight)
                        dev_losses.append((len(dev_batch), dev_loss.item()))
                    dev = sum(count * value for count, value in dev_losses) / len(dev_rows)
                row = {"step": step, "loss": loss.item(), "training_ce": reconstruction.item(),
                       "dev_ce": dev, "elapsed_s": round(perf_counter() - start, 2)}
                history.append(row)
                print(mode, row, flush=True)
                if dev < best:
                    best, best_step = dev, step
                    torch.save({"config": asdict(config), "vocabulary": vocabulary.items,
                                "mode": mode, "step": step, "state": model.state_dict()},
                               directory / f"{mode}.pt")
            if step == args.steps:
                break
    checkpoint = torch.load(directory / f"{mode}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, {"best_step": best_step, "model_parameters": sum(p.numel() for p in model.parameters()),
                   "training_s": round(perf_counter() - start, 2), "history": history}


def score(rows):
    per_strategy = {}
    for strategy in STRATEGIES:
        subset = [row for row in rows if row["expected"] == strategy]
        per_strategy[strategy] = {
            "valid": sum(row["status"] == "valid" for row in subset),
            "correct": sum(row["strategy"] == strategy for row in subset),
            "attempts": len(subset),
        }

    def correct(row):
        return row["status"] == "valid" and row["strategy"] == row["expected"]

    groups = list(batches(rows, 4))
    return {
        "goals": len(groups),
        "paired_correct": sum(correct(group[0]) and correct(group[2]) for group in groups),
        "all_renderings_correct": sum(all(map(correct, group)) for group in groups),
        "valid": sum(row["status"] == "valid" for row in rows),
        "attempts": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "per_strategy": per_strategy,
    }


@torch.inference_mode()
def evaluate(args, model, vocabulary, goals, examples, mode, condition):
    rows, cases, positive, negative = [], [], [], []
    start = perf_counter()
    donors = list(range(len(goals)))
    if condition == "transferred":
        # Each destination receives the two strategies from a different template.
        for i, spec in enumerate(goals):
            candidates = [*range(i + 1, len(goals)), *range(i)]
            donors[i] = next(j for j in candidates if goals[j].family != spec.family)
    for goal_indices in batches(list(range(len(goals))), args.batch_goals):
        specs = [goals[i] for i in goal_indices]
        batch = [(examples[i][0], examples[donors[i]][1]) for i in goal_indices]
        goal, proofs = collate(batch)
        if mode == "label":
            latent = model.encode_label(proofs.shape[0], proofs.device)
        else:
            latent = model.encode_proof(proofs)
        if condition == "zero":
            latent = torch.zeros_like(latent)
        else:
            proof_indices = torch.arange(proofs.shape[0])
            flat = latent.flatten(1)
            positive.extend(F.cosine_similarity(flat, flat[proof_indices ^ 1]).tolist())
            negative.extend(F.cosine_similarity(flat, flat[proof_indices ^ 2]).tolist())
        predictions = model.generate(goal, latent, BOS, EOS, vocabulary.forbidden, args.max_tokens)
        for i, spec, group in zip(goal_indices, specs, batches(predictions, 4)):
            source = goals[donors[i]].id if condition != "zero" and mode != "label" else None
            for index, prediction in enumerate(group):
                proof = vocabulary.decode(prediction)
                rows.append({"goal_id": spec.id, "family": spec.family, "expected": STRATEGIES[index // 2],
                             "latent_source_goal_id": source,
                             "rendering": index % 2, "proof": proof})
                cases.append((spec.statement, proof or ""))
    generation_s = perf_counter() - start
    verdicts = check(args.lean, args.output, cases, args.verify_workers)
    if any(verdict.status == "checker_error" for verdict in verdicts):
        error = next(v for v in verdicts if v.status == "checker_error")
        raise RuntimeError(f"Lean evaluation infrastructure failed: {error.message}")
    for row, verdict in zip(rows, verdicts):
        row.update(asdict(verdict))
        if row["proof"] is None:
            row["status"] = "no_eos"
    result = score(rows)
    result["decode_s"] = round(generation_s, 2)
    result["verify_s"] = round(perf_counter() - start - generation_s, 2)
    if positive:
        result["within_strategy_cosine"] = sum(positive) / len(positive)
        result["between_strategy_cosine"] = sum(negative) / len(negative)
    return result, rows


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, choices=SPLIT_COUNTS, default=24)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--latent-slots", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=320)
    p.add_argument("--validate-every", type=int, default=40)
    p.add_argument("--batch-goals", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--contrastive-weight", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--verify-workers", type=int, default=4)
    p.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    p.add_argument("--transfer", action="store_true", help="Also swap latents across test templates.")
    p.add_argument("--allow-local", action="store_true", help="Allow development outside Slurm.")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not args.allow_local and not os.environ.get("SLURM_JOB_ID"):
        p.error("Training requires a Slurm allocation (--allow-local for local development).")
    limit = int(os.environ.get("SLURM_CPUS_PER_TASK", str(len(os.sched_getaffinity(0)))))
    if not 0 < args.threads <= limit or not 0 < args.verify_workers <= limit:
        p.error(f"Thread and worker counts must each fit the {limit} allocated CPUs.")
    if min(args.width, args.latent_dim, args.latent_slots, args.heads, args.layers, args.steps,
           args.validate_every, args.batch_goals, args.max_tokens) < 1 or args.width % args.heads:
        p.error("Dimensions and step counts must be positive and width must be divisible by heads.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    goals = generate(args.size)
    split_counts = write_dataset(args.output / "dataset.jsonl", goals)
    references = [(g.statement, proof) for g in goals for proof in g.proofs]
    reference_batch = min(96, max(24, len(references) // args.verify_workers))
    verdicts = check(args.lean, args.output, references, args.verify_workers, reference_batch)
    for index, verdict in enumerate(verdicts):
        expected = STRATEGIES[index % 4 // 2]
        if (verdict.status, verdict.strategy) != ("valid", expected):
            raise RuntimeError(f"Invalid reference {goals[index // 4].id}, proof {index % 4}: {verdict}")
    verify_references_s = round(perf_counter() - start, 2)
    print(f"Lean approved {len(references)} reference proofs in {verify_references_s}s: {split_counts}", flush=True)

    splits = {split: [goal for goal in goals if goal.split == split] for split in split_counts}
    vocabulary = Vocabulary(splits["train"])
    encoded = {split: [(vocabulary.encode(g.statement), [vocabulary.encode(p) for p in g.proofs])
                       for g in split_goals] for split, split_goals in splits.items()}
    train_length = max(len(t) for goal, proofs in encoded["train"] for t in [goal, *proofs])
    max_length = max(args.max_tokens + 1, train_length)
    if any(len(t) > max_length for dataset in encoded.values() for goal, proofs in dataset
           for t in [goal, *proofs]):
        p.error("An evaluation sequence exceeds the configured context. Increase --max-tokens.")
    config = ModelConfig(vocab=len(vocabulary.items), max_length=max_length, width=args.width,
                         latent_dim=args.latent_dim, slots=args.latent_slots, layers=args.layers,
                         heads=args.heads)
    summary = {"settings": vars(args) | {"output": str(args.output)}, "torch": torch.__version__,
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "split_counts": split_counts,
               "reference_proofs_verified": len(references), "reference_verify_s": verify_references_s,
               "model": asdict(config), "runs": {}}
    for mode in args.modes:
        model, metrics = train(args, mode, config, encoded["train"], encoded["dev"], args.output, vocabulary)
        state = metrics | {"evaluation": {}}
        summary["runs"][mode] = state
        for split in ("seen", "test"):
            evals = state["evaluation"][split] = {}
            conditions = ["encoded"]
            if mode == "strategy":
                conditions.append("zero")
                if args.transfer and split == "test":
                    conditions.append("transferred")
            for condition in conditions:
                result, rows = evaluate(args, model, vocabulary, splits[split], encoded[split], mode, condition)
                evals[condition] = result
                write_json(args.output / f"{mode}-{split}-{condition}.json", rows)
                print(f"{mode} {split} {condition}: {result}", flush=True)
                write_json(args.output / "summary.json", summary)
    summary["elapsed_s"] = round(perf_counter() - start, 2)
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
