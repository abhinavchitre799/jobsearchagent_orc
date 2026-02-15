## Evals (Sample)

This folder contains a small, pragmatic eval harness for the two core generators:
- hiring manager message (`outputType: "message"`)
- cover letter (`outputType: "cover-letter"`)

### Prereqs
- API + worker running (so `POST /generate` works)
- `OPENAI_API_KEY` configured

### Run
```bash
python3 evals/run_generation_sample.py
```

Optional:
```bash
python3 evals/run_generation_sample.py --api-base http://localhost:8000
python3 evals/run_generation_sample.py --cases evals/cases/generation_sample.jsonl
```

### What It Checks
- Output length caps (word count)
- Must-include / must-not-include tokens (e.g. no placeholders)
- Grounding marker presence: `"(from resume)"`
- Message-specific: greeting, manager name near top, company mentioned
- Cover-letter-specific: `Sincerely`, candidate name, company mentioned, paragraph count

This is intentionally a *starting point* (cheap, fast, reproducible). For higher fidelity, add an LLM-judge rubric on top.

