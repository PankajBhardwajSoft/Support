"""Step 6: rebuild every demonstration transcript, or check that the saved ones still match.

    python scripts/run_all_transcripts.py           rewrite transcripts/*.txt
    python scripts/run_all_transcripts.py --check   compare only; exit 1 if any differs
    python scripts/run_all_transcripts.py step5_cache.txt   only the transcripts named

Every script runs under MOCK_LLM with the network blocked (scripts/_bootstrap.py).
Only stdout is saved. Random IDs, timestamps and timings are ignored when
comparing. The pytest transcripts (fresh-venv runs) and the Groq transcripts
(real model, PANKAJ's PC) are not rebuilt here.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (command after "python", transcript), in the order the steps were built.
RUNS = [
    (["scripts/smoke_test.py"], "step1_smoke_test.txt"),
    (["dataset.py"], "step2_dataset.txt"),
    (["scripts/build_index.py"], "step2_build_index.txt"),
    (["scripts/calibrate_threshold.py"], "step2_calibration.txt"),
    (["scripts/demo_grounded.py", "kb_fixed"], "step2_grounded_demo_kb_fixed.txt"),
    (["scripts/demo_grounded.py", "kb_sentence"], "step2_grounded_demo_kb_sentence.txt"),
    (["scripts/eval_retrieval.py"], "step2_retrieval_eval.txt"),
    (["scripts/escalation_threshold.py"], "step3_escalation_threshold.txt"),
    (["scripts/demo_crew.py"], "step3_crew.txt"),
    (["scripts/demo_memory.py"], "step3_memory.txt"),
    (["scripts/demo_guardrails.py"], "step3_guardrails.txt"),
    (["scripts/demo_api.py"], "step4_api.txt"),
    (["scripts/run_eval.py"], "step4_eval.txt"),
    (["scripts/demo_review.py"], "step5_review.txt"),
    (["scripts/demo_governance.py"], "step5_governance.txt"),
    (["scripts/demo_cache.py"], "step5_cache.txt"),
]

_VOLATILE = [
    (re.compile(r"\b[0-9a-f]{32}\b"), "<id>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T[\d:.]+\+00:00"), "<ts>"),
    (re.compile(r'"duration_ms": [\d.]+'), '"duration_ms": <n>'),
    (re.compile(r"time: \d+ ms"), "time: <n> ms"),
]


def _comparable(text: str) -> list[str]:
    """Drop the header line, warning lines (stderr in older transcripts) and volatile values."""
    lines, skip_next = [], False
    for line in text.splitlines()[1:]:
        if skip_next and line.startswith("  "):
            skip_next = False
            continue
        skip_next = False
        if "Warning:" in line:
            skip_next = True
            continue
        for pattern, repl in _VOLATILE:
            line = pattern.sub(repl, line)
        lines.append(line)
    return lines


def run(args: list[str]) -> tuple[str, int]:
    env = {k: v for k, v in os.environ.items() if k != "MOCK_LLM"}  # always the mock
    env["TQDM_DISABLE"] = "1"
    done = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return done.stdout, done.returncode


def main(argv: list[str]) -> int:
    check = "--check" in argv
    names = [a for a in argv if a.endswith(".txt")]
    runs = [r for r in RUNS if not names or r[1] in names]
    failed = []
    for args, name in runs:
        path = os.path.join(ROOT, "transcripts", name)
        with open(path, encoding="utf-8") as f:
            saved = f.read()
        out, code = run(args)
        fresh = f"{saved.splitlines()[0]}\n{out}exit code: {code}\n"
        if check:
            same = _comparable(fresh) == _comparable(saved)
            print(f"{'same   ' if same else 'DIFFERS'}  {name}")
            if not same:
                failed.append(name)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fresh)
            print(f"wrote  {name}  (exit code {code})")
        if code != 0:
            failed.append(f"{name} (exit code {code})")
    print(f"\n{len(runs)} transcripts, {len(failed)} problem(s){': ' + ', '.join(failed) if failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
