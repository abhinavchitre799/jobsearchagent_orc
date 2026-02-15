## Job Search Agent (OpenAI RAG + Queue-Orchestrated Agents)

Generate a LinkedIn outreach message or cover letter grounded in your resume. The system uses OpenAI embeddings + chat with a FastAPI orchestrator service and queue-backed agent workers (retrieve → plan → draft → critique → revise).

### OpenAI UI (queue-backed runtime)
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```
- Configure env (recommended: repo-local `.env`, which is gitignored):
  ```bash
  cat > .env <<'EOF'
  OPENAI_API_KEY=
  # SERPAPI_API_KEY=
  # JOB_DISCOVERY_PROVIDER=serpapi  # optional; auto-selects serpapi if SERPAPI_API_KEY is set
  # REDIS_URL=redis://localhost:6380/0
  EOF
  ```
- Quick dev start (starts Redis + worker + API + UI server):
  ```bash
  ./scripts/dev_up.sh
  # UI:  http://localhost:5500/index.html
  ```
- Stop dev processes:
  ```bash
  ./scripts/dev_down.sh
  ```
- Start one worker process per agent queue (recommended):
  ```bash
  python worker.py --queues retriever
  python worker.py --queues planner
  python worker.py --queues drafter
  python worker.py --queues critic
  python worker.py --queues reviser
  python worker.py --queues job_discovery
  ```
- Run the orchestrator API server:
  ```bash
  uvicorn api:app --reload --port 8000
  ```
- Serve the static UI:
  ```bash
  python -m http.server 5500
  # then open http://localhost:5500/index.html
  ```
- In the UI, fill in the fields, upload/paste your resume, and click **Generate drafts**. You can copy or download the outputs. The page calls `http://localhost:8000/generate` and shows a loader overlay while the agent runs.
- The UI now includes a **Discover PM Openings** panel. Enter a natural language prompt (default: `Fetch all PM jobs in US`), fetch jobs, then choose **Use this job**. The generate panel appears with role/company/hiring-manager/JD prefilled.

### OpenAI RAG + LLM CLI
- Requires `pip install openai` and `OPENAI_API_KEY` set.
- Example:
  ```bash
  export OPENAI_API_KEY=sk-...
  python agent.py --resume resume.txt \
    --input-file jd.txt \
    --name "Your Name" \
    --role "Data Scientist" \
    --company "Acme" \
    --hiring-manager "Jordan" \
    --hm-note "Thanks for reaching out about the platform PM opening."
  ```
- Flags: `--embedding-model text-embedding-3-small` (default), `--chat-model gpt-4o-mini` (default), `--top-k 3` snippets, `--hm-note` to blend a LinkedIn note with the JD, `--output-type message|cover-letter` (default message), `--orchestrate` to enable the agentic loop.

### Testing
- After installing requirements, run:
  ```bash
  pytest
  ```
  Tests use a mocked OpenAI client to cover chunking, retrieval, and the API endpoint without network calls.

### Quick start
- Put your resume in plain text (e.g., `resume.txt`). If you have a PDF, export it to text first.
- Run:
  ```bash
  python agent.py --resume resume.txt \
    --input-file jd.txt \
    --role "Data Scientist" \
    --company "Acme" \
    --hiring-manager "Jordan"
  ```
  Or pass inline text instead of a file:
  ```bash
  python agent.py --resume resume.txt \
    --input-text "Saw your post about the ML Engineer role at Acme..." \
  ```

### What it does
- Ingests your resume, splits it into readable chunks, embeds them, and matches against the JD/LinkedIn message via cosine similarity.
- Selects top resume snippets, then drafts either a LinkedIn-ready note or a cover letter with a chat model, grounded in those snippets.
- Uses queued agent workers for retrieval/planning/drafting/critique/revision, coordinated by the FastAPI orchestrator.

### Architecture (current)
1. **Orchestrator API (`api.py`)**: validates input, builds run state, and controls agent switching.
2. **Queue layer (`queueing.py`)**: enqueues jobs and waits for results via Redis + RQ.
3. **Specialized workers (`worker_tasks.py`)**:
   - `retriever`: embedding-based chunk retrieval
   - `planner`: plan generation
   - `drafter`: draft generation
   - `critic`: JSON critique
   - `reviser`: critique-driven revision
   - `job_discovery`: LLM prompt parsing + SerpAPI fetch + strict verification + pagination
4. **Queue-aware orchestration (`orchestrator_queue.py`)**:
   - **Fixed path**: retrieve → plan → draft → critique → optional revise
   - **Adaptive path**: orchestrator chooses next action each cycle
5. **Guardrails**:
   - Revision cap (max 2)
   - No-improvement stop
   - Deterministic fallback draft if the loop stalls

### Options
- `--resume` (required): Path to your resume text file.
- `--input-text` or `--input-file` (required): JD or recruiter message to match against.
- `--name` (optional): Name for the intro (defaults to "Candidate").
- `--role`, `--company`, `--hiring-manager`: Optional context to personalize the opener.
- `--top-k`: Number of resume snippets to ground with (default 3).
- `--orchestrate`: Enable the orchestration loop.

### API contract (POST /generate)
- Required: `resumeText`, `jdText`
- Optional: `name`, `hmNote`, `role`, `company`, `hiringManager`
- Controls: `topK`, `embeddingModel`, `chatModel`, `outputType`, `orchestrate`
- Behavior: request stays synchronous but internally dispatches queued tasks to worker processes.

### API contract (POST /jobs/discover)
- Required: `prompt`
- Optional: `page`, `pageSize`, `country`, `strict` (strict is enforced `true` in current MVP)
- Returns: `count`, `page`, `pageSize`, `hasNextPage`, `source`, and verified `jobs`.
- Error semantics:
  - `503` if `SERPAPI_API_KEY` is missing while provider is `serpapi`
  - `502` for worker/provider failures
  - `504` for discovery timeouts

### Example output
```
Hi Jordan,
I'm Alex and I'm excited about the Data Scientist role at Acme. I pulled a few highlights that line up with what you're looking for:
- Built end-to-end churn prediction models in Python, improving retention by 9% and shipping dashboards in Streamlit for product stakeholders.
- Productionized ETL pipelines on Airflow with unit tests and monitoring to keep latency under 5 minutes for downstream analysts.
- Partnered with PMs to design A/B tests and translate results into roadmap decisions for a 10M user base.
Would you be open to a quick chat this week? Thanks for your time!
```
