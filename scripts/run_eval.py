"""Task 13: score 15 queries for Accuracy, Grounding, Completeness and Safety with the mock judge."""

import _bootstrap  # noqa: F401

from ola_agent.evaluation import TEST_SET, averages, build_judge_prompt, evaluate, judge


def main():
    rows = evaluate()

    print("Judge prompt, filled in for query 1:")
    print(build_judge_prompt(TEST_SET[0], rows[0].answer, rows[0].sources, rows[0].chunks))
    print()

    print(f"{'#':>2}  {'kind':<12} {'expected':<8} {'top source':<10} {'Acc':>5} {'Grnd':>5} {'Comp':>5} {'Safe':>5}  question")
    for i, r in enumerate(rows, 1):
        s = r.scores
        print(f"{i:>2}  {r.case.kind:<12} {r.case.expected_doc or '-':<8} {(r.sources or ['-'])[0]:<10} "
              f"{s.accuracy:>5.2f} {s.grounding:>5.2f} {s.completeness:>5.2f} {s.safety:>5.2f}  {r.case.question}")
    avg = averages(rows)
    print(f"{'':>2}  {'AVERAGE':<12} {'':<8} {'':<10} {avg['accuracy']:>5.2f} {avg['grounding']:>5.2f} "
          f"{avg['completeness']:>5.2f} {avg['safety']:>5.2f}")
    print()
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r.scores.reason}")
    print()

    # Negative control: the judge must score a wrong answer lower, or it proves nothing.
    wrong = rows[11].answer  # the data-retention answer, given to the Tier 2 question
    control = judge(TEST_SET[2], wrong, ["KB-12"], rows[2].chunks)
    print(f"Negative control (KB-12 text as the answer to the Tier 2 question): {control.model_dump()}")
    checks = {
        "15 queries scored, 12 in scope and 3 out of scope": len(rows) == 15
        and sum(r.case.kind == "out_of_scope" for r in rows) == 3,
        "every topic KB-01 to KB-12 covered": {r.case.expected_doc for r in rows} - {None} == {f"KB-{n:02d}" for n in range(1, 13)},
        "judge scores a wrong answer lower": control.accuracy == 0 and control.completeness == 0 and control.grounding < 1,
    }
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    ok = all(checks.values())
    print("EVALUATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
