# TODO (next session or Mariam)

Status after session 2 (2026-10-12): everything a sandbox session can do is done. The
recommender, interview chatbot (with the LLM feedback path), API, front end, 46 tests,
family-rule evaluation on 50 queries, hand-judged subset on 10 queries, notebook (executed
once, outputs cleared), Dockerfile, README and WALKTHROUGH are in place and `make check` is
green. What is left needs a machine with internet access or Mariam's own judgement:

1. **MiniLM numbers.** `pip install -r requirements-minilm.txt`, let it download
   `sentence-transformers/all-MiniLM-L6-v2` once, run `make eval`; paste the three `minilm`
   rows into both README tables and adjust the discussion. (Hugging Face is blocked from the
   sandbox: CONNECT 403, checked again on 2026-10-12.)
2. **Real postings.** Download a permissively licensed job-postings CSV locally (Kaggle search:
   "job postings"), write a 20-line mapper to `id,title,company,location,description`, run
   `JOBREC_POSTINGS=path make eval` and report the numbers separately, labelled as real data.
   Keep the synthetic CSV as the default so CI stays offline. (Kaggle also blocked.)
3. **Docker.** `docker build -t jobrec . && docker run -p 8000:8000 jobrec`, then
   `curl localhost:8000/health`; not possible in the sandbox (no daemon).
4. **Re-judge `data/eval/judged.csv`.** The 255 judgements are one reader's opinion (see
   `data/eval/JUDGING.md`). Overrule any row you disagree with, keep the reason honest, run
   `make eval` and `python scripts/build_notebooks.py --execute`, and update the README table
   if a number moves.
5. **One live LLM run.** Set `JOBREC_LLM_PROVIDER=http` with any OpenAI-compatible endpoint in
   `.env`, answer three interview questions, and check the page shows the reworded feedback
   with `source` ≠ `rule`. Only mock endpoints were used so far.

Optional extension (see WALKTHROUGH "One thing you could extend"): a seniority term in the
score, measured on the judged subset.

Not planned: persistent sessions, authentication, deployment. The repo is a demo.
