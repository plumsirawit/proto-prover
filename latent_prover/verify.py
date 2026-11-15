"""Check the pilot's restricted tactic language and inspect Lean proof terms."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory


HEADER = """import Lean
open Lean Elab Command in
elab "inspect_pilot " label:ident : command => do
  let name := (← getCurrNamespace) ++ label.getId
  let some (.thmInfo info) := (← getEnv).find? name
    | throwError "missing theorem {name}"
  let axioms ← collectAxioms name
  if axioms.any (fun ax => !([`propext, `Classical.choice, `Quot.sound].contains ax)) then
    throwError "unexpected axioms in {name}: {axioms}"
  let constants := info.value.getUsedConstants
  let monotone := constants.contains `Nat.add_le_add
  let gap := constants.contains `Nat.exists_eq_add_of_le && constants.contains `Exists.casesOn
  let strategy := if gap then "gap_witness" else if monotone then "monotonicity" else "other"
  logInfo m!"PILOT_VERDICT {label.getId} {strategy}"
set_option maxHeartbeats 200000
run_cmd Lean.logInfo m!"PILOT_READY"
"""

TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9.]*|[0-9]|:=|[^\s]")
VERDICT = re.compile(r"^PILOT_VERDICT case_(\d+) (\w+)$", re.MULTILINE)
DIAGNOSTIC = re.compile(r"Trial\.lean:(\d+):\d+: error: ([^\n]*)")
TACTICS = {"exact", "have", "obtain", "rcases", "rw", "omega"}
TERMS = {"Nat.add_le_add", "Nat.le_trans", "Nat.le_refl", "Nat.exists_eq_add_of_le",
         "_", "with", "(", ")", "[", "]", "⟨", "⟩", ",", ":", ":=", "≤"}
LOCALS = re.compile(r"(?:[ahjbm]|bound|gap|delta|eqn|step)[0-2](?:_2)?\Z")


def safe_proof(body):
    """Accept only known tactics and Lean tokens in an indented tactic block."""
    lines = body.split("\n")
    if len(body) > 3000 or len(lines) > 16:
        return False
    for line in lines:
        tokens = TOKEN.findall(line)
        if not tokens or tokens[0] not in TACTICS:
            return False
        if any(token not in TERMS and LOCALS.fullmatch(token) is None for token in tokens[1:]):
            return False
    return True


@dataclass(frozen=True)
class Verdict:
    status: str
    strategy: str | None = None
    message: str | None = None


def check_batch(lean, scratch, cases, timeout):
    results = [Verdict("outside_grammar") for _ in cases]
    blocks = [HEADER]
    eligible = {}
    next_line = HEADER.count("\n") + 2
    for idx, (statement, body) in enumerate(cases):
        if not safe_proof(body):
            continue
        eligible[idx] = next_line
        tactic_block = "\n  ".join(line.strip() for line in body.split("\n"))
        # Separate namespaces prevent a later candidate from citing earlier results.
        blocks.append(f"namespace trial_{idx}\ntheorem case_{idx} {statement} := by\n"
                      f"  {tactic_block}\ninspect_pilot case_{idx}\nend trial_{idx}\n")
        next_line += blocks[-1].count("\n") + 1
    if not eligible:
        return results
    with TemporaryDirectory(dir=scratch, prefix="lean-") as directory:
        path = Path(directory, "Trial.lean")
        path.write_text("\n".join(blocks), encoding="utf-8")
        try:
            checked = subprocess.run([lean, "-j", "1", str(path)], cwd=directory,
                                     capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            for idx in eligible:
                results[idx] = Verdict("checker_error", message=str(error))
            return results
    output = checked.stdout + checked.stderr
    if checked.returncode < 0 or "PILOT_READY" not in output.splitlines():
        for idx in eligible:
            results[idx] = Verdict("checker_error", message=output[-1000:])
        return results
    errors = [(int(line), message) for line, message in DIAGNOSTIC.findall(output)]
    for (idx, start), end in zip(eligible.items(), [*list(eligible.values())[1:], next_line]):
        message = next((msg for line, msg in errors if start <= line < end), None)
        results[idx] = Verdict("lean_rejected", message=message)
    for index, strategy in VERDICT.findall(output):
        if results[int(index)].message is None:
            results[int(index)] = Verdict("valid", strategy)
    return results


def check(lean, scratch, cases, workers, batch_size=24):
    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    if workers == 1:
        return [v for batch in batches for v in check_batch(lean, scratch, batch, 90)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check_batch, lean, scratch, batch, 90) for batch in batches]
        return [v for future in futures for v in future.result()]
