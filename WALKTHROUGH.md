# Walkthrough (for Mariam)

## Why it is built this way

A recommender you can explain beats one you cannot. Every ranking here is two numbers you can
recompute by hand: cosine similarity between two texts, and the fraction of a posting's required
skills that appear in the CV. The explanation ("matches X, missing Y") is literally the second
number. The LLM is optional and off for the same reason: the demo must run on a laptop with no
key, and nothing a professor sees should depend on a third-party API.

The dataset is synthetic because the sandbox could not download a public one. The generator is
in the repo and the README says so up front; never let anyone think these are real postings.
The loader needs five columns, so swapping in a Kaggle CSV is a local job.

## File map

- `scripts/generate_postings.py` — twelve role families, each with a required-skill pool, a
  nice-to-have pool, and work sentences; seeded, so `--check` proves the CSV is reproducible.
- `src/jobrec/skills.py` — the skill dictionary (`resources/skills.json`) and `extract_skills`,
  whole-phrase matching with longest-phrase-first ordering.
- `src/jobrec/embed.py` — three backends behind one `fit/transform` interface: TF-IDF (default),
  IDF-weighted mean GloVe, MiniLM (optional). All rows are unit-norm so cosine is a dot product.
- `src/jobrec/recommend.py` — the blended score and the matched/missing/bonus partition.
- `src/jobrec/interview.py` — `build_questions` (intro, per-skill, gap, behavioural from
  `resources/questions.json`), `score_answer` (the 60/25/15 rule), `InterviewSession`.
- `src/jobrec/providers/` — `LLMProvider` protocol, `NullProvider`, `HttpChatProvider` for any
  OpenAI-compatible endpoint; every failure returns `None` so the rule-based path takes over.
- `src/jobrec/api.py` + `static/index.html` — FastAPI with Pydantic models, one HTML page.
- `scripts/evaluate.py`, `data/eval/` — 50 hand-written CVs, the family relevance rule, P@k/MRR.

## Three questions a professor may ask

**1. "Your precision is 0.98. Why should I believe that?"**
You should not, as a real-world number. The postings are generated from the same skill
vocabulary the queries use, so a word-overlap model is close to the ceiling by construction.
The useful comparisons are relative: α = 1 (skills only) is worse than text because of ties,
and mean-pooled GloVe alone collapses onto a few "hub" postings. The honest next step is the
same pipeline on real postings with a human-judged label subset, which is TODO item 1 and 3.

**2. "Why TF-IDF in 2026? Why not a sentence transformer?"**
Both sit behind the same interface; MiniLM is implemented but could not be downloaded where
this was built. For 500 documents, TF-IDF with bigrams is fast, needs no model file, and you
can print the terms that drove a match. Sentence embeddings win when CV and posting say the
same thing in different words, which this synthetic data rarely does. The GloVe row shows the
cost of a weak semantic embedding without the skill signal. Say "I would measure it" rather
than guess where MiniLM lands.

**3. "The interview scorer counts keywords. Isn't that gameable?"**
Yes, and the README says it is a rehearsal aid. The rule is deliberately visible: coverage of
expected terms, length up to 60 words, a bonus for a concrete example. A candidate who games
it by listing terms still ends up with the vocabulary an interviewer expects. Grading
correctness needs a human or an LLM judge, and an LLM judge brings back the dependency the
design avoids by default; the provider interface is where it would plug in.

## One thing you could extend

Add a `feedback` path to the provider: after the rule-based score, let the LLM rewrite the
comment in plain language and name one thing to say next time, with the score unchanged.
About 30 lines in `interview.py`; the rule stays the source of truth, and it demonstrates
"LLM as presenter, not as judge".
