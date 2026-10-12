# Hand-judged relevance subset (`judged.csv`)

`labels.csv` marks a posting relevant to a query when the two share a role family. That rule
is auditable but blunt: it counts a Lead role shown to a fresh graduate as a hit, and a good
cross-family match (a data-platform CV ranked against SRE roles) as a miss. `judged.csv` is a
second, smaller label set made by reading each posting against the CV.

## What was judged

- **Queries:** 10 of the 50 (`Q02 Q07 Q11 Q16 Q18 Q22 Q32 Q39 Q48 Q50`). They were chosen on
  purpose to include the CVs that straddle families (incubator firmware + Flutter, ML at an
  IoT startup, data platform with Kubernetes, perception on embedded targets, ML across embedded
  and data) plus a few plain ones. The subset is therefore *harder* than the full set, and its
  numbers are not comparable with the 50-query table.
- **Pairs:** the union of the top-10 lists of TF-IDF and GloVe at α ∈ {0, 0.5, 1} for each of
  those queries: 255 (query, posting) pairs, 19–32 per query. `scripts/build_judging_pool.py`
  rebuilds that pool and `--check` (run by `make check`) fails if a pair in a current top-10
  has no judgement, so the file cannot silently go stale.
- A pair that is not in `judged.csv` was never looked at. The evaluation counts such a pair as
  not relevant and reports the number as `unjudged` (0 for the committed code and data).

## Criteria

Each pair gets one of three values, with a one-line reason in the `reason` column:

| judgement | meaning |
|---|---|
| `relevant` | a recruiter would send this posting to this candidate: the role is what the CV describes, the candidate has most of the required skills, and the seniority is within reach |
| `arguable` | a plausible sideways move or a seniority mismatch: an adjacent role whose required skills the CV largely covers (e.g. SRE for a data-platform CV with Kubernetes and monitoring), or the right role at Senior/Lead level for a CV that reads as graduate or single-project |
| `wrong` | the candidate would not apply: a different field, or both the field and the seniority are off |

Seniority rule used throughout: a CV that says "recent graduate", "junior", or describes one
project with no years counts as early-career, so Senior (4–5 y) and Lead (6–7 y) postings in
the right family are `arguable`, not `relevant`. A CV in a professional voice with no years
stated is matched to Senior postings as `relevant` and to Lead postings as `arguable`.

Two readings are reported: **strict** (only `relevant` counts) and **lenient** (`relevant`
or `arguable`). The truth is somewhere between them.

## Who judged, and the limits

The judgements were made in the same build session that wrote the 50 CV queries and the
posting generator, by reading each posting's title, seniority, required and nice-to-have
skills against the CV. They are one reader's opinion, not a recruiter panel, and the reader is
not independent of the data. Mariam should overrule any row she disagrees with (edit the
`judgement` column, keep the reason honest, re-run `make eval`). Because the postings are
synthetic and built from 12 role families, the judgement mostly separates *seniority* and
*adjacent role* cases that the family rule cannot see; it says nothing about how either
label set would behave on real postings.
