# job-recommender-chatbot

This repo shows a small, inspectable job recommender: paste CV text, get the top-k postings
with the skills that matched and the skills that are missing, then rehearse a mock interview
for the posting you pick. Everything runs offline with no API key; an LLM backend can be
switched on behind an interface but is off by default.

**The postings are synthetic.** The build sandbox could not reach Kaggle or the Hugging Face
Hub, so `data/postings.csv` (500 rows) is produced by `scripts/generate_postings.py` from
twelve role-family templates with a fixed seed. Company names are fictional. The loader
accepts any CSV with `id,title,company,location,description` columns, so a real dataset
can be dropped in locally (see TODO.md).

## Data path

```mermaid
flowchart LR
  CV[CV text] --> S[skills.py<br/>dictionary extraction]
  CV --> E[embed.py<br/>TF-IDF · GloVe · MiniLM]
  P[(postings.csv)] --> E
  P --> S
  E --> R[recommend.py<br/>cosine + skill coverage]
  S --> R
  R --> X[top-k + matched / missing skills]
  X --> I[interview.py<br/>question bank · rule-based score]
  I -.optional.-> L[providers/<br/>LLM follow-up, off by default]
  R & I --> A[api.py FastAPI] --> H[static/index.html]
```

Score per posting: `(1 − α) · cosine(CV, posting) + α · coverage`, where coverage is the share
of the posting's required skills found in the CV by the dictionary extractor. α = 0.5 by
default; the evaluation below compares α ∈ {0, 0.5, 1}.

## Run it

```bash
pip install -r requirements.txt
make check                      # ruff + 46 pytest tests + data/label freshness + quick eval
make serve                      # http://localhost:8000  (API docs at /docs)
make vectors && make eval       # optional: GloVe backend (66 MB download) + full evaluation
pip install nbclient ipykernel && python scripts/build_notebooks.py --execute   # run the notebook once
docker build -t jobrec . && docker run -p 8000:8000 jobrec   # not run in the sandbox (no daemon)
```

Example API call:

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"cv_text":"ESP32 firmware in C with FreeRTOS, MQTT over TLS, Python with pandas","k":3}'
```

returns, per hit, `score`, `similarity`, `skill_coverage`, `matched_skills`, `missing_skills`,
`bonus_skills` (nice-to-haves the CV has) and a one-line `explanation`. `POST /interview/start`
with a `posting_id` and the CV text opens a session; `POST /interview/answer` scores each answer
and returns the next question, then a summary after the last one.

## What the outputs look like

For a CV mentioning ESP32, C, FreeRTOS, MQTT, Flutter, Firebase, pandas and scikit-learn the
top hits (TF-IDF, α = 0.5) are IoT-engineer and ML-engineer postings, e.g.
`IoT Solutions Engineer — matches c, esp32, mqtt, python; missing linux, tls`. The interview for
that posting asks eight questions: an intro, one per required skill (a skill-specific question
for 23 skills, e.g. MQTT QoS/retained/last-will, otherwise a category template), one
"gap" question about a missing skill, and one behavioural question. Each answer gets a 0–100
score (60 % expected-keyword coverage, 25 % length up to 60 words, 15 % for a concrete example),
the keywords it missed, a one-line tip ("Next time, mention qos and give one concrete example
with a number.") and, below 75, a follow-up question. With an LLM provider switched on
(`JOBREC_LLM_PROVIDER=http`, any OpenAI-compatible endpoint) one call per answer may reword the
comment, the tip and the follow-up; the score and the keyword lists always come from the rule,
and the response says `source: "http"` so the page can show which wording came from where.
Any provider failure or unparseable reply falls back to the rule text.

## Evaluation (numbers from `make eval`, in `figures/eval_results.json`)

`data/eval/queries.csv` holds 50 CV texts written by hand for this repo, 4–5 per role family,
some phrased with aliases or without exact skill names. A posting counts as relevant when its
role family equals the family the CV was written for (`scripts/build_eval_labels.py`; on
average 41.7 relevant postings per query, chance P@k = 0.083).

| embedder | α | P@1 | P@5 | P@10 | MRR |
|---|---|---|---|---|---|
| tfidf | 0.0 | 1.00 | 0.99 | 0.98 | 1.00 |
| tfidf | 0.5 | 0.98 | 0.98 | 0.97 | 0.98 |
| tfidf | 1.0 | 0.94 | 0.94 | 0.91 | 0.96 |
| glove | 0.0 | 0.60 | 0.57 | 0.53 | 0.68 |
| glove | 0.5 | 0.96 | 0.96 | 0.93 | 0.97 |
| glove | 1.0 | 0.94 | 0.94 | 0.91 | 0.96 |

![precision@5 per embedder and alpha](figures/eval_p5.png)

What the data supports:

- **The benchmark is easy, and that is a property of the synthetic data, not of the model.**
  Postings are built from the same skill vocabulary the CVs use, so word-level TF-IDF is near
  the ceiling. Do not read 0.98 as a real-world number.
- **Skill coverage alone (α = 1) is worse than text similarity** because many postings tie on
  coverage; the text breaks the ties. α = 0.5 loses little and buys the explanation.
- **Averaged GloVe vectors alone are poor (P@5 0.57).** Mean pooling blurs specific terms, and
  two short mobile-developer postings become "hubs" that appear for CVs of almost every family.
  Adding skill coverage fixes most of it (0.96), which is the case for keeping the explicit skill
  signal even with a semantic embedder.
- The one TF-IDF miss at α = 0.5 is a data-platform CV (Kubernetes, Prometheus, CI/CD) ranked
  against cloud/DevOps postings; by the family rule that is wrong, by a human reading it is
  arguable. The label rule is the weak point of this evaluation (see Limitations).

### Hand-judged subset

The family rule is blunt, so 10 of the 50 queries were re-labelled by reading each posting
against the CV (`data/eval/judged.csv`, criteria and who judged in `data/eval/JUDGING.md`). The
10 were chosen to include the CVs that straddle families (incubator firmware + Flutter, ML at
an IoT startup, data platform with Kubernetes, ML across embedded and data), so this subset is
harder than the full set. Every top-10 pair of every configuration was judged (255 pairs:
159 relevant, 54 arguable, 42 wrong); `make check` fails if a code change moves an unjudged
posting into a top-10. "Strict" counts only `relevant`; "lenient" also counts `arguable`.

| embedder | α | P@10 family rule | P@10 strict | P@10 lenient |
|---|---|---|---|---|
| tfidf | 0.0 | 0.98 | 0.83 | 0.99 |
| tfidf | 0.5 | 0.88 | 0.74 | 0.96 |
| tfidf | 1.0 | 0.77 | 0.66 | 0.88 |
| glove | 0.0 | 0.58 | 0.52 | 0.72 |
| glove | 0.5 | 0.82 | 0.71 | 0.92 |
| glove | 1.0 | 0.77 | 0.66 | 0.88 |

![precision@10 on the judged subset](figures/eval_judged_p10.png)

- **The rule and the judgement never flatly contradict each other.** No pair the rule calls
  relevant was judged `wrong`, and no pair it rejects was judged `relevant`; all 54
  disagreements are `arguable`, split evenly: 27 the rule accepts and a human doubts, 27 the
  rule rejects and a human would keep.
- **The recommender is blind to seniority.** All 27 rule-accepted-but-arguable pairs are
  Senior or Lead postings shown to CVs that read as graduate or single-project. Nothing in the
  score looks at `years_min`, and the strict column pays for it.
- **Skill coverage pulls in adjacent roles.** At α = 0.5 the data-platform CV (Q18) gets 9
  cloud/SRE postings in its top 10, every one requiring exactly the four skills the extractor
  found (Spark, Kubernetes, monitoring, CI/CD); a human calls 8 of them arguable, the rule calls
  all 9 wrong. The same mechanism puts two software-tester postings in the backend CV's (Q22)
  top 10. That is the cost of a coverage term over a small skill vocabulary.
- Within a column the ordering of configurations is the same as on the full set, so the
  relative conclusions above stand; the absolute numbers on this subset are lower because the
  subset was chosen to be hard.

The MiniLM sentence-embedding backend (`requirements-minilm.txt`) is implemented but **has no
numbers here**: the model could not be downloaded in the sandbox. TODO.md item 1.

## Limitations

- Synthetic postings (see above); fictional companies; English only.
- Skill extraction is a dictionary of 91 skills with aliases, matched as whole phrases. It has
  no context: "C" matches the letter in "section C", "Go" is only recognised as "golang", and a
  skill spelt differently from every alias is missed. Its coverage bounds the explanations.
- Relevance labels come from a rule (same role family) for 50 queries, and from one reader's
  judgement for 10 of them (`JUDGING.md`); that reader is the author of the queries, not a
  recruiter, so the judged numbers are an opinion, not ground truth.
- The ranking ignores seniority: a Lead posting and a Junior one with the same skills score
  the same for every CV.
- Interview scoring is keyword and length based. It rewards mentioning the right terms, not
  being right. It is a rehearsal aid, not an assessment. An LLM provider can reword the
  feedback but was only exercised against a mock endpoint in the tests, never a live model.
- Interview sessions live in process memory (capped at 200); restarting the server drops them.
- Docker image not built in the sandbox (no daemon); MiniLM backend not exercised.

## Next steps

See TODO.md: real postings dataset through the same loader, MiniLM numbers, a Docker build
check, a seniority term in the score (`years_min` against years stated in the CV), and a second
judge for `judged.csv`.

## Related

This recommender is a from-scratch rebuild of a student-project idea; the skill vocabulary
is weighted towards embedded/IoT/ML roles because that is the author's field. The MQTT, ESP32
and FreeRTOS interview questions are the ones an IoT firmware interview would actually ask.

## Licence

MIT, see LICENSE. GloVe vectors (if downloaded) are public domain (PDDL), from the gensim-data
release.
