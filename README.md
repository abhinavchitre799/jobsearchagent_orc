## Job Search Agent (OpenAI RAG)

Generate a LinkedIn outreach note grounded in your resume. Both CLI and browser UI use OpenAI embeddings + chat (no offline mode).

### OpenAI UI
- Install deps and run the API server:
  ```bash
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...
  uvicorn api:app --reload --port 8000
  ```
- Serve the static UI:
  ```bash
  python -m http.server 5500
  # then open http://localhost:5500/index.html
  ```
- In the UI, fill in the fields, choose **LinkedIn message** or **Cover letter**, upload/paste your resume, and click **Generate message**. You can copy or download the draft. The page calls `http://localhost:8000/generate`.

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
- Flags: `--embedding-model text-embedding-3-small` (default), `--chat-model gpt-4o-mini` (default), `--top-k 3` snippets, `--hm-note` to blend a LinkedIn note with the JD, `--output-type message|cover-letter` (default message).

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
    --name "Your Name" \
    --role "Data Scientist" \
    --company "Acme" \
    --hiring-manager "Jordan"
  ```
  Or pass inline text instead of a file:
  ```bash
  python agent.py --resume resume.txt \
    --input-text "Saw your post about the ML Engineer role at Acme..." \
    --name "Your Name"
  ```

### What it does
- Ingests your resume, splits it into readable chunks, embeds them, and matches against the JD/LinkedIn message via cosine similarity.
- Selects top resume snippets, then drafts either a LinkedIn-ready note or a cover letter with a chat model, grounded in those snippets.

### Options
- `--resume` (required): Path to your resume text file.
- `--input-text` or `--input-file` (required): JD or recruiter message to match against.
- `--name` (required): Your name for the intro.
- `--role`, `--company`, `--hiring-manager`: Optional context to personalize the opener.
- `--top-k`: Number of resume snippets to ground with (default 3).

### Example output
```
Hi Jordan,
I'm Alex and I'm excited about the Data Scientist role at Acme. I pulled a few highlights that line up with what you're looking for:
- Built end-to-end churn prediction models in Python, improving retention by 9% and shipping dashboards in Streamlit for product stakeholders.
- Productionized ETL pipelines on Airflow with unit tests and monitoring to keep latency under 5 minutes for downstream analysts.
- Partnered with PMs to design A/B tests and translate results into roadmap decisions for a 10M user base.
Would you be open to a quick chat this week? Thanks for your time!
```
