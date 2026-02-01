## Quick orientation for AI coding agents

This repo implements a small RAG-powered outreach generator (LinkedIn message / cover letter) using OpenAI embeddings + chat. The agent should focus on correctness, minimal surface changes, and preserving the strict safety/prompt invariants found in `prompts.py`.

Key artifacts:
- `agent.py` — CLI tool. Ingests a resume + JD (file or inline) and prints a generated message. Look here for flag semantics and token-estimate heuristics.
- `api.py` — FastAPI server that exposes POST /generate. See `GenerateRequest`/`GenerateResponse` models for the exact JSON contract.
- `rag.py` — Core logic: chunking, embedding cache, retrieval, planning, critique/revise cycle, and orchestration (`generate_agentic_message` and `generate_orchestrated_message`). This is the most important file to read for end-to-end behavior.
- `prompts.py` — System/user prompts. These encode strict rules (e.g. "use ONLY the resume snippets" and every referenced achievement must include “(from resume)”). Preserve these rules when changing outputs or tests.
- `tests/` — Offline tests using a `FakeOpenAI` client (see `tests/test_rag.py`). Tests exercise chunking, retrieval, API wiring and orchestration stop conditions.

Core architecture / data flow (big picture):
- Input: resume text + job description / hiring manager note.
- Preprocessing: `split_into_chunks` splits the resume into paragraph-like chunks (~900 chars by default).
- Embeddings: `embed_texts` calls OpenAI embeddings and caches results in `_EMBED_CACHE` keyed by (model, sha1(text)). Small, in-memory cache only.
- Retrieval: `retrieve_chunks_with_embeddings` scores chunks by cosine similarity and applies `LOW_SCORE_THRESHOLD` (0.08) and `MAX_TOP_K` (10).
- Generation: `build_llm_message` composes system + user prompt (from `prompts.py`) and calls `client.chat.completions.create` to get the output.
- Orchestration: `generate_orchestrated_message` runs a loop (plan → draft → critique → revise → finalize) using `decide_next_action` (LLM-driven). There are safe fallbacks — deterministic draft or early finalize — if the orchestrator becomes stagnant.

Project-specific conventions and important gotchas
- Prompt safety is enforced in `prompts.py`. System prompts require exact behaviours (e.g., add "(from resume)" to referenced achievements). Don't relax or reword these rules lightly — they are tested by the suite.
- Embedding cache is process-local. Tests rely on monkeypatching `rag.embed_texts`/`api.OpenAI` to avoid network calls.
- Default model strings live in `agent.py`/`api.py` flags and are used widely (`text-embedding-3-small`, `gpt-4o-mini`). Change model strings carefully.
- Chunking is paragraph-aware (see `split_into_chunks`). When adding sample resumes to tests, prefer multi-paragraph text so chunking behaviour is exercised.
- Retrieval filters out low-similarity snippets using `LOW_SCORE_THRESHOLD`. If tests fail due to too few snippets, tweak top_k in test inputs rather than lowering the threshold globally.

Developer workflows / commands (reproducible locally)
- Install deps:
  pip install -r requirements.txt
- Run API server (FastAPI + Uvicorn):
  export OPENAI_API_KEY=sk-...
  uvicorn api:app --reload --port 8000
- Serve the static UI (for manual browser testing):
  python -m http.server 5500
  # open http://localhost:5500/index.html (page posts to /generate)
- CLI usage (example):
  python agent.py --resume resume.txt --input-file jd.txt --name "Your Name"
- Run tests (uses offline FakeOpenAI):
  pytest

Integration points & how tests stub them
- OpenAI client: code creates `OpenAI(api_key=...)` from the `openai` package. In tests, `api.OpenAI` and `rag.embed_texts` are monkeypatched to return deterministic embeddings and completions (`tests/test_rag.py`). When adding tests, follow the same pattern.
- HTTP contract: `POST /generate` expects JSON with `resumeText` and `jdText`. The endpoint will raise 400 if either is empty and 500 if `OPENAI_API_KEY` is not configured.

Places to edit when adding features
- If you change how chunks are summarized or how many snippets are cited, update `rag.summarize_chunk` and `USER_PROMPT_TEMPLATE` in `prompts.py`.
- If you adjust orchestration rules (allowed actions), modify `ORCHESTRATOR_SYSTEM_PROMPT` and corresponding logic in `rag.generate_orchestrated_message` and ensure tests cover stagnation and revision caps.

Examples to reference in code review suggestions
- Prefer small, additive changes: e.g., add a new CLI flag in `agent.py`, pass it through to `rag.generate_*` functions, and add test coverage that monkeypatches `rag.embed_texts` like the existing tests.
- Any change to `prompts.py` should include a test asserting the revised prompt still enforces the "(from resume)" rule where applicable.

If anything above is ambiguous or you'd like more detail, tell me which aspect (prompts, orchestration loop, tests, API contract, or embedding/retrieval) and I will expand or iterate.
