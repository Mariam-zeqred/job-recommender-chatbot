# Walkthrough (for Mariam)

## Why it is built this way

A recommender you can explain beats one you cannot. Every ranking is two numbers you can
recompute by hand: cosine similarity between two texts, and the fraction of a posting's required
skills that appear in the CV. The explanation ("matches X, missing Y") is literally the second
number. The LLM is optional and off for the same reason: the demo must run on a laptop with no
key, and nothing a professor sees should depend on a third-party API. Where the LLM is allowed
in, it presents and never judges: the rule scores first, the LLM only rewords.

The dataset is synthetic because the sandbox could not download a public one; the generator is
in the repo and the README says so up front.

## File map

- `scripts/generate_postings.py` — twelve role families, seeded; `--check` proves the CSV
  reproducible.
- `src/jobrec/skills.py` — the skill dictionary (`resources/skills.json`) and `extract_skills`.
- `src/jobrec/embed.py` — TF-IDF (default), IDF-weighted GloVe, MiniLM (optional), one interface.
- `src/jobrec/recommend.py` — the blended score and the matched/missing/bonus partition.
- `src/jobrec/interview.py` — `build_questions`, `score_answer` (the 60/25/15 rule), `rule_tip`,
  `parse_review` (the three labelled lines an LLM must return), `InterviewSession.review`.
- `src/jobrec/providers/` — `NullProvider` default, `HttpChatProvider` for any OpenAI-compatible
  endpoint; every failure returns `None`.
- `src/jobrec/api.py` + `static/index.html` — FastAPI, Pydantic models, one HTML page.
- `scripts/evaluate.py`, `data/eval/` — 50 hand-written CVs, the family rule (`labels.csv`),
  the hand-judged subset (`judged.csv`, criteria in `JUDGING.md`), P@k/MRR, two figures.
- `scripts/build_judging_pool.py` — rebuilds the 255 judged pairs; `make check` fails if a code
  change moves an unjudged posting into a top-10.

## Three questions a professor may ask

**1. "Your precision is 0.98. Why should I believe that?"**
As a real-world number, you should not. The postings are generated from the same skill
vocabulary the queries use, so a word-overlap model is near the ceiling by construction. What
the evaluation does show is relative: skills-only ranking loses to text because of ties, and
mean-pooled GloVe alone collapses onto hub postings. The hand-judged subset adds what the
family rule could not see: the ranking ignores seniority, so Lead postings reach graduate CVs.
Name the next step yourself: real postings and a second judge.

**2. "Why TF-IDF in 2026? Why not a sentence transformer?"**
Both sit behind the same interface; MiniLM is implemented but could not be downloaded where
this was built. For 500 documents, TF-IDF with bigrams is fast, needs no model file, and shows
the terms that drove a match. Sentence embeddings win when CV and posting say the
same thing in different words, which this synthetic data rarely does. Say "I would measure it"
rather than guess where MiniLM lands.

**3. "The interview scorer counts keywords. Isn't that gameable?"**
Yes, and the README says it is a rehearsal aid. The rule is deliberately visible: coverage of
expected terms, length up to 60 words, a bonus for a concrete example. Grading correctness
needs a human or an LLM judge, and an LLM judge brings back the dependency the design avoids.
The feedback path is the compromise: the LLM may explain the rule's verdict in better words
and suggest what to say next time, but `score` never comes from it, and the response says
which wording came from where.

## One thing you could extend

Add a seniority term: parse "N years" from the CV (default 0 for a graduate) and penalise
postings whose `years_min` exceeds it by more than two. About 20 lines in `recommend.py` and
one column in the evaluation table; the judged subset already contains the 27 pairs that would
move, so you can measure the effect before and after.
