"""Task 6: print the escalation-score formula, its distribution on our dataset, and the threshold."""

import _bootstrap  # noqa: F401

from dataset import SUPPORT_TICKETS
from ola_agent.tools.ticket_lookup import (
    ESCALATED_WEIGHT, ESCALATION_THRESHOLD, MAX_DAYS, RECENCY_WEIGHT, THRESHOLD_PERCENTILE, escalation_score,
)


def percentile(sorted_values, p):
    # Nearest rank, lower: index floor(p/100 * (n - 1)).
    return sorted_values[int(p / 100 * (len(sorted_values) - 1))]


def main():
    print(f"escalation_score = {ESCALATED_WEIGHT} * escalated + {RECENCY_WEIGHT} * (days_since_created / {MAX_DAYS})")
    rows = sorted(((escalation_score(t["escalated"], t["days_since_created"]), t) for t in SUPPORT_TICKETS),
                  key=lambda r: r[0])
    scores = [s for s, _ in rows]
    print(f"\n{len(scores)} tickets, {sum(t['escalated'] for _, t in rows)} with escalated=True")
    print("Sorted scores:", ", ".join(f"{s:.3f}" for s in scores))
    print("\nPercentiles (nearest rank):")
    for p in (50, 70, 75, 80, 90):
        print(f"  p{p} = {percentile(scores, p):.3f}")
    measured = percentile(scores, THRESHOLD_PERCENTILE)
    print(f"\nThreshold = p{THRESHOLD_PERCENTILE} = {measured:.3f} (code constant {ESCALATION_THRESHOLD:.2f})")
    max_plain = max(s for s, t in rows if not t["escalated"])
    min_esc = min(s for s, t in rows if t["escalated"])
    print(f"Highest never-escalated ticket: {max_plain:.3f}; lowest escalated ticket: {min_esc:.3f}")

    flagged = [(s, t) for s, t in rows if s >= ESCALATION_THRESHOLD]
    print(f"\nFlagged (score >= {ESCALATION_THRESHOLD:.2f}): {len(flagged)} tickets")
    for s, t in flagged:
        print(f"  {t['record_id']}  score {s:.3f}  escalated={t['escalated']!s:5}  days={t['days_since_created']:2}  status={t['status']}")
    plain = [t["record_id"] for s, t in flagged if not t["escalated"]]
    print(f"Flagged without escalated=True (age alone): {plain}")

    ok = abs(measured - ESCALATION_THRESHOLD) < 1e-9 and bool(plain)
    print("\nThreshold matches the measured percentile and age alone can flag a ticket:", "PASS" if ok else "FAIL")
    print("ESCALATION THRESHOLD:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
