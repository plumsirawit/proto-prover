# Can a proof latent control Lean’s choice of strategy?

The latent changed which route the decoder attempted. Neither the 0.90M nor the
3.51M model produced both strategies for any of the 240 held-out template goals
at the 2,000-goal scale. The oracle-label controls had the same failure.

The test holds the Lean theorem and decoding settings fixed and swaps the
continuous latent between two verified proofs. A theorem counts as a paired
success if Lean accepts both outputs and each uses the corresponding strategy.
The rendering check requires the same result from both source renderings of
each strategy.

The original Apache-2.0 dataset uses inequalities over sums of natural numbers.
One route composes the input inequalities using monotonicity. The other extracts
additive gap witnesses, rewrites, and closes the resulting arithmetic goal.
Each strategy has two tactic scripts. Training, validation, and test use separate
expression templates. An additional `seen` split contains new goals of training
templates. Lean checks the fixed theorem, rejects unapproved transitive axioms,
and reads strategy indicators from the elaborated proof term.

The standard model has width 128 and two layers each in the proof encoder and
decoder. It encodes each proof as four 32-dimensional vectors. The decoder
receives those vectors and the Lean statement. The strategy model maps either
rendering to a canonical training target and learns to separate opposite
strategies for the same goal.
Controls use exact reconstruction, canonical targets alone where indicated, or
the oracle strategy label. All training uses CPU PyTorch on Euler Slurm with a
ceiling of 8 CPUs and 24 GiB per run.

### Results

Each cell shows paired successes on familiar templates, then on held-out templates.

| Total goals | Width | Exact reconstruction | Canonical targets | Contrastive latent | Oracle label |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 | 128 | 1/2, 0/4 | — | 0/2, 0/4 | 0/2, 0/4 |
| 200 | 128 | 0/24, 0/24 | 1/24, 0/24 | 2/24, 0/24 | 0/24, 0/24 |
| 2,000 | 128 | 13/240, 0/240 | 19/240, 0/240 | 15/240, 0/240 | 28/240, 0/240 |
| 2,000 | 256 | — | — | 7/240, 0/240 | 11/240, 0/240 |

Lean 4.19.0 verified all 96, 800, and 8,000 reference scripts, respectively.
At 200 goals, removing the latent yielded 84 accepted outputs on held-out
templates, all using gap witnesses. At 2,000 goals, the contrastive model
at width 128 attempted the requested route in all 960 held-out decodes, as
measured by the signature lemma in the text. Lean accepted 456/480 gap-witness
attempts at width 128 and 450/480 at width 256. The oracle labels reached
480/480 and 440/480, respectively. At both widths, the contrastive and oracle
models each failed all 480 held-out monotonicity attempts.

Using latents from a different template produced identical scripts in 938/960
cases at width 128 and 960/960 at width 256. Both models retained their original
verification counts. The decoder appears to use the latent mainly to select a
route. For example, it attempted to prove
`((1 + a0) + a1) ≤ ((1 + b0) + b1)` with a monotonicity term for
`((a1 + a0) + k) ≤ ((b1 + b0) + k)`, which Lean rejected as a type mismatch.

The study uses one seed, two strategy types, synthetic statements, and a small
tactic grammar. The primary evaluations start with an encoded reference proof
or an oracle label. A useful next experiment is to give the decoder the Lean
expression tree and let it assemble a checkable proof plan. The persistent
failure with oracle labels makes that a more direct test of the observed problem.

### Runs

Slurm jobs `20951137`, `20951328`, `20951422`, and `20951798` all completed.
They allocated eight CPUs and 24 GiB each. The largest reported step MaxRSS
was 2.16 GiB. Both 2,000-goal jobs used byte-identical datasets, 1,800 updates per
mode, and the same latent dimensions. They used PyTorch `2.8.0+cpu` and
selected checkpoints by validation cross-entropy. Datasets, checkpoints, source
checksums, and per-candidate verdicts live under `$SCRATCH/proto-prover` in the
directories named for the dataset size and Slurm job ID.
