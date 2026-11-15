"""Train a plain causal Transformer to complete a Lean goal with a proof."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import Goal, SPLIT_COUNTS
from .experiment import EOS, PAD, Vocabulary, batches, tokens, write_json
from .verify import check


class GoalTransformer(nn.Module):
    def __init__(self, vocabulary_size, width, layers, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=PAD)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            width, 4, 2 * width, dropout=0.05, batch_first=True, norm_first=True,
        ) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary_size)
        self.register_buffer("causal", torch.ones(max_length, max_length, dtype=torch.bool).triu(1),
                             persistent=False)

    def forward(self, ids):
        length = ids.shape[1]
        x = self.embedding(ids) + self.position(torch.arange(length, device=ids.device))
        for block in self.blocks:
            x = block(x, src_mask=self.causal[:length, :length], is_causal=True)
        return self.output(self.norm(x))

    def generate(self, goals, vocabulary, max_tokens):
        start = torch.tensor([len(goal) for goal in goals])
        cursor = start.clone()
        longest = int(start.max())
        result = torch.full((len(goals), longest + max_tokens), PAD, dtype=torch.long)
        for index, goal in enumerate(goals):
            result[index, :len(goal)] = goal
        rows = torch.arange(len(goals))
        finished = torch.zeros(len(goals), dtype=torch.bool)
        for offset in range(max_tokens):
            logits = self(result[:, :longest + offset])[rows, cursor - 1]
            logits[:, vocabulary.forbidden] = -torch.inf
            next_token = torch.where(finished, EOS, logits.argmax(dim=-1))
            result[rows, cursor] = next_token
            cursor += 1
            finished |= next_token.eq(EOS)
            if finished.all():
                break
        return [vocabulary.decode(row[a:b].tolist()) for row, a, b in zip(result, start, cursor, strict=True)]


def examples(goals, vocabulary, mode):
    for goal in goals:
        prefix = vocabulary.encode(goal.statement)
        proofs = goal.proofs if mode == "mixed" else goal.proofs[:1]
        for proof in proofs:
            sequence = torch.cat((prefix, vocabulary.encode(proof)[1:]))
            targets = sequence[1:].clone()
            # The final goal token predicts the first proof token. Earlier
            # positions provide context without contributing to the loss.
            targets[:len(prefix) - 1] = PAD
            yield sequence[:-1], targets


def batch_loss(model, rows):
    inputs, targets = zip(*rows, strict=True)
    targets = pad_sequence(targets, batch_first=True)
    logits = model(pad_sequence(inputs, batch_first=True))
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD, reduction="sum")
    return loss, targets.ne(PAD).sum()


def fit(args, model, train, dev, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    history = []
    best_loss, best_step = float("inf"), 0
    start = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(len(train), (args.batch_size,)).tolist()
        optimizer.zero_grad(set_to_none=True)
        loss, count = batch_loss(model, [train[i] for i in indices])
        train_loss = loss / count
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step in (1, args.steps) or step % args.validate_every == 0:
            model.eval()
            with torch.no_grad():
                dev_loss, total = 0.0, 0
                for batch in batches(dev, args.batch_size):
                    batch_sum, count = batch_loss(model, batch)
                    dev_loss += float(batch_sum)
                    total += int(count)
            dev_loss /= total
            row = {"step": step, "train_ce": float(train_loss.detach()), "dev_ce": dev_loss,
                   "elapsed_s": round(perf_counter() - start, 2)}
            history.append(row)
            print(output.name, row, flush=True)
            if dev_loss < best_loss:
                best_loss, best_step = dev_loss, step
                torch.save(model.state_dict(), output / "best.pt")
    torch.save(model.state_dict(), output / "last.pt")
    return {"best_step": best_step, "history": history, "train_s": round(perf_counter() - start, 2)}


@torch.inference_mode()
def evaluate(args, model, goals, vocabulary, path):
    start = perf_counter()
    predictions = []
    for batch in batches(goals, args.batch_size):
        predictions.extend(model.generate([vocabulary.encode(g.statement) for g in batch], vocabulary, args.max_tokens))
    decode_s = perf_counter() - start
    checked = check(args.lean, args.output,
                    [(g.statement, p or "") for g, p in zip(goals, predictions, strict=True)], args.verify_workers)
    rows = []
    for goal, proof, verdict in zip(goals, predictions, checked, strict=True):
        if verdict.status == "checker_error":
            raise RuntimeError(f"Lean verification failed: {verdict.message}")
        row = {"goal_id": goal.id, "family": goal.family, "proof": proof, **asdict(verdict)}
        if proof is None:
            row["status"] = "no_eos"
        row["matches_reference"] = proof is not None and any(tokens(proof) == tokens(p) for p in goal.proofs)
        rows.append(row)
    write_json(path, rows)
    return {"accepted": sum(r["status"] == "valid" for r in rows), "goals": len(rows),
            "matches_reference": sum(r["matches_reference"] for r in rows),
            "strategies": dict(Counter(r["strategy"] for r in rows if r["status"] == "valid")),
            "statuses": dict(Counter(r["status"] for r in rows)), "decode_s": round(decode_s, 2),
            "verify_s": round(perf_counter() - start - decode_s, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, choices=(2000,), required=True)
    p.add_argument("--dataset", type=Path, required=True, help="Trusted dataset.jsonl from the original run.")
    p.add_argument("--checkpoint-run", type=Path,
                   help="Evaluate final checkpoints from a trusted completed run on the test split.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--verify-workers", type=int, required=True)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--modes", nargs="+", choices=("mixed", "monotonicity"), default=("mixed", "monotonicity"))
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("This experiment must run in a Slurm allocation.")
    limit = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 0 < args.threads <= limit or not 0 < args.verify_workers <= limit:
        p.error("Thread and worker counts must fit the Slurm allocation.")
    if min(args.width, args.layers, args.steps, args.validate_every, args.batch_size, args.max_tokens) < 1 or args.width % 4:
        p.error("Dimensions and counts must be positive and width must be divisible by four.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    payload = args.dataset.read_bytes()
    goals = [Goal(**json.loads(line)) for line in payload.splitlines()]
    splits = {name: [g for g in goals if g.split == name] for name in SPLIT_COUNTS[args.size]}
    if {name: len(goals) for name, goals in splits.items()} != SPLIT_COUNTS[args.size]:
        p.error("The archived dataset does not have the expected splits.")
    vocabulary = Vocabulary(splits["train"])
    max_length = 256
    if any(len(tokens(g.statement)) + 2 + max(args.max_tokens, *(len(tokens(p)) + 1 for p in g.proofs))
           > max_length for g in goals):
        p.error("A sequence exceeds the experiment's 256-token limit.")
    summary = {"slurm_job": os.environ["SLURM_JOB_ID"], "dataset_sha256": sha256(payload).hexdigest(),
               "settings": vars(args) | {"dataset": str(args.dataset), "output": str(args.output),
                                         "checkpoint_run": str(args.checkpoint_run) if args.checkpoint_run else None},
               "model": {"width": args.width, "layers": args.layers, "max_length": max_length,
                         "vocabulary": vocabulary.items}, "runs": {}}
    if args.checkpoint_run:
        previous = json.loads((args.checkpoint_run / "summary.json").read_text(encoding="utf-8"))
        if ("elapsed_s" not in previous or previous["dataset_sha256"] != summary["dataset_sha256"]
                or previous["model"] != summary["model"] or previous["settings"]["max_tokens"] != args.max_tokens):
            p.error("A completed run with matching dataset, model, and token budget is required.")
        summary["checkpoint_job"] = previous["slurm_job"]
    for mode in args.modes:
        torch.manual_seed(args.seed)
        output = args.output / mode
        output.mkdir()
        model = GoalTransformer(len(vocabulary.items), args.width, args.layers, max_length)
        if args.checkpoint_run:
            checkpoint = args.checkpoint_run / mode / "last.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            score = evaluate(args, model.eval(), splits["test"], vocabulary, output / "test-final.json")
            summary["runs"][mode] = {"checkpoint_step": previous["settings"]["steps"], "test_final": score}
            print(mode, "test_final", score, flush=True)
            continue
        train = list(examples(splits["train"], vocabulary, mode))
        dev = list(examples(splits["dev"], vocabulary, mode))
        result = fit(args, model, train, dev, output)
        result.update({"parameters": sum(p.numel() for p in model.parameters()), "train_examples": len(train),
                       "evaluation": {}})
        summary["runs"][mode] = result
        model.eval()
        if result["best_step"] != args.steps:
            result["evaluation"]["train_final"] = evaluate(
                args, model, splits["train"], vocabulary, output / "train-final.json",
            )
        model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True))
        for name in ("train", "seen", "test"):
            result["evaluation"][name] = evaluate(args, model, splits[name], vocabulary, output / f"{name}.json")
            print(mode, name, result["evaluation"][name], flush=True)
            write_json(args.output / "summary.json", summary)
    summary["elapsed_s"] = round(perf_counter() - start, 2)
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
