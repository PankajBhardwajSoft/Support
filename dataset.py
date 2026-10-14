"""Seeded, deterministic support-ticket dataset for the Ola support agent (Part 1, Task 1).

Run:  python dataset.py

Design choices (also stated in README.md so the dataset can be reproduced):
- SEED = 42, N_RECORDS = 50, record IDs TKT-0001 .. TKT-0050.
- Category and status weights are in CATEGORY_WEIGHTS and STATUS_WEIGHTS below.
- resolution_time_hours: 1.0 to 168.0 hours. Simple account fixes close within
  an hour, while defect and billing investigations can take up to a week.
- days_since_created: integer 0 to 30.
- escalated: True with probability ESCALATED_PROBABILITY; any ticket whose status
  is "Escalated" is also set to escalated=True so records never contradict.
- No record is ever edited by hand. If a threshold check fails, change SEED or
  the weights and regenerate.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

SEED = 42
N_RECORDS = 50

CATEGORY_WEIGHTS = {
    "Billing": 0.25,
    "Technical Issue": 0.25,
    "Account Access": 0.20,
    "Product Defect": 0.15,
    "General Inquiry": 0.15,
}

STATUS_WEIGHTS = {
    "Open": 0.20,
    "In Progress": 0.25,
    "Escalated": 0.10,
    "Resolved": 0.25,
    "Closed": 0.20,
}

RESOLUTION_HOURS_RANGE = (1.0, 168.0)
DAYS_SINCE_CREATED_RANGE = (0, 30)
ESCALATED_PROBABILITY = 0.15

MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
ESCALATED_PCT_RANGE = (10.0, 30.0)


def generate_tickets(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    categories = list(CATEGORY_WEIGHTS)
    statuses = list(STATUS_WEIGHTS)
    tickets = []
    for i in range(1, n + 1):
        category = rng.choices(categories, weights=list(CATEGORY_WEIGHTS.values()))[0]
        status = rng.choices(statuses, weights=list(STATUS_WEIGHTS.values()))[0]
        resolution_time_hours = round(rng.uniform(*RESOLUTION_HOURS_RANGE), 1)
        days_since_created = rng.randint(*DAYS_SINCE_CREATED_RANGE)
        escalated = rng.random() < ESCALATED_PROBABILITY or status == "Escalated"
        tickets.append(
            {
                "record_id": f"TKT-{i:04d}",
                "category": category,
                "status": status,
                "resolution_time_hours": resolution_time_hours,
                "days_since_created": days_since_created,
                "escalated": escalated,
            }
        )
    return tickets


def validate(tickets: list[dict]) -> list[tuple[str, bool]]:
    """Return (check description, passed) for every structural threshold in the brief."""
    checks = []
    checks.append((f"at least 40 records (got {len(tickets)})", len(tickets) >= 40))
    cat_counts = Counter(t["category"] for t in tickets)
    for cat in CATEGORY_WEIGHTS:
        checks.append((f"category {cat!r} >= {MIN_PER_CATEGORY} (got {cat_counts[cat]})", cat_counts[cat] >= MIN_PER_CATEGORY))
    status_counts = Counter(t["status"] for t in tickets)
    for st in STATUS_WEIGHTS:
        checks.append((f"status {st!r} >= {MIN_PER_STATUS} (got {status_counts[st]})", status_counts[st] >= MIN_PER_STATUS))
    pct = escalated_pct(tickets)
    lo, hi = ESCALATED_PCT_RANGE
    checks.append((f"escalated% in [{lo:g}, {hi:g}] (got {pct:.1f}%)", lo <= pct <= hi))
    return checks


def escalated_pct(tickets: list[dict]) -> float:
    return 100.0 * sum(t["escalated"] for t in tickets) / len(tickets)


SUPPORT_TICKETS = generate_tickets()


def main() -> int:
    tickets = SUPPORT_TICKETS
    print(f"SUPPORT_TICKETS: {len(tickets)} records (seed={SEED})\n")
    print("Count per category:")
    for cat, count in Counter(t["category"] for t in tickets).most_common():
        print(f"  {cat:<16} {count}")
    print("\nCount per status:")
    for st, count in Counter(t["status"] for t in tickets).most_common():
        print(f"  {st:<16} {count}")
    n_esc = sum(t["escalated"] for t in tickets)
    print(f"\nEscalated: {n_esc}/{len(tickets)} = {escalated_pct(tickets):.1f}%\n")
    print("Threshold checks:")
    checks = validate(tickets)
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    all_ok = all(ok for _, ok in checks)
    print("\nDATASET:", "PASS" if all_ok else "FAIL (change SEED or weights; never hand-edit records)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
