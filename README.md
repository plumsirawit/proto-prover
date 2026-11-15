# proto-prover

An experiment on learning continuous representations of Lean proof strategies.
The proof encoder compresses a complete tactic script into a fixed number of
vectors. The decoder receives the theorem and those vectors, then writes a new
proof. The pilot tests whether changing the vectors switches the strategy for
the same theorem. This experiment uses known proof latents. Generation from a
bare goal and latent diffusion are future experiments.

See [REPORT.md](REPORT.md) for the findings from the CPU runs.

The original dataset contains natural-number inequalities with two proofs each:
assembling inequalities by monotonicity, and constructing additive witnesses for
the gaps before rewriting. Each route has two Lean renderings. Validation and
test expression templates are held out from training. A separate `seen` split
uses new goals made from training templates. Both goals and proofs are generated
under this repository's Apache-2.0 license; there are no corpus downloads.

The experiment compares exact autoencoding, canonical proof decoding with a
contrastive loss, and a decoder given the correct strategy label. The optional
`canonical` mode isolates the effect of canonicalization by omitting the
contrastive term. The contrastive model is also evaluated with its latent zeroed. Lean
checks all references and predictions in temporary files, inspects transitive
axioms, and uses proof-term dependencies as indicators of the two routes. The
checker accepts a small tactic vocabulary specific to this pilot.

### Running on Euler

Use a CPU PyTorch environment on scratch (the experiment used PyTorch
`2.8.0+cpu` and Python 3.10). Install the Lean version recorded in
`lean-toolchain`. The job requests 8 cores and 3 GiB per core. Set `run_root` to
an existing directory on scratch and `lean_exe` to the pinned Lean executable.

```bash
uv venv "$run_root/venv" --python /usr/bin/python3 --no-cache
python_exe="$run_root/venv/bin/python"
uv pip install --python "$python_exe" --cache-dir "$run_root/uv-cache" \
  --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu

sbatch --output="$run_root/slurm-%j.log" scripts/experiment.sbatch \
  "$PWD" "$run_root" "$python_exe" "$lean_exe" 24 \
  --steps 320 --width 128 --latent-slots 4 --latent-dim 32
```

Valid dataset sizes are `24`, `200`, and `2000`. The directory named
`<size>-<job-id>` contains the source snapshot and checksums, datasets, weights,
per-candidate Lean results, and `results/summary.json`. Compare
`paired_correct` for both strategies on each goal and `all_renderings_correct`
for the stricter result that requires both renderings of each strategy to work.
`--seed`, `--steps`, `--width`, `--latent-slots`, and `--latent-dim` can be set
independently. `--transfer` also tests latents taken from a different goal and
expression template. The script stages no large files in the repository.

To compare the expression-tree decoder against an archived width-128
contrastive run, point to that run's trusted results directory. The comparison
uses its dataset and checkpoint, retrains only a linear route selector, and
replays the original decoder on the same 240 test goals.

```bash
sbatch --output="$run_root/slurm-%j.log" scripts/experiment.sbatch \
  "$PWD" "$run_root" "$python_exe" "$lean_exe" 2000 \
  --structured --baseline "$run_root/2000-20951422/results"
```

Lean exports the elaborated goal and local inequality hypotheses. The planner
decomposes addition and finds paths through those hypotheses. It renders the
path either as monotonicity or as additive gap witnesses. The comparison saves
both decoders' verdicts and a control that tries both plans without a latent.

To run the plain goal-to-proof Transformer on the same archived split, pass
the original dataset file. The input sequence contains the goal tokens followed
by proof tokens. The loss scores only the proof tokens. At evaluation the model
starts with the goal and greedily emits one proof, which Lean checks.

```bash
sbatch --output="$run_root/slurm-%j.log" scripts/experiment.sbatch \
  "$PWD" "$run_root" "$python_exe" "$lean_exe" 2000 \
  --goal-to-proof --dataset "$run_root/2000-20951422/results/dataset.jsonl"
```

The default compares training on all four proof scripts per goal with training
on one canonical monotonicity proof per goal. Checkpoints and per-goal results
are saved on scratch. The summary includes Lean acceptance on training goals,
new goals from familiar templates, and the 240 held-out template goals.
Pass `--checkpoint-run` with the results directory of a completed run to score
its final checkpoints on the held-out split without retraining.
