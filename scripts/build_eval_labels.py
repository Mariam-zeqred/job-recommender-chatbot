"""Write data/eval/labels.csv from the relevance rule.

Relevance rule: a posting is relevant to a query when its `role_family` equals the family
the query's CV was written for (the `role_family` column of queries.csv, chosen by the
author when the 50 CV texts were written). The rule is deliberately simple and auditable;
its limits are discussed in README "Evaluation". Run with --check to verify the committed
file.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"


def build() -> str:
    queries = list(csv.DictReader((EVAL / "queries.csv").open(encoding="utf-8")))
    postings = list(csv.DictReader((ROOT / "data" / "postings.csv").open(encoding="utf-8")))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["query_id", "posting_id"])
    for q in queries:
        for p in postings:
            if p["role_family"] == q["role_family"]:
                w.writerow([q["id"], p["id"]])
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = build()
    out = EVAL / "labels.csv"
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"{out} is out of date; run scripts/build_eval_labels.py", file=sys.stderr)
            return 1
        print(f"{out}: up to date")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({text.count(chr(10)) - 1} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
