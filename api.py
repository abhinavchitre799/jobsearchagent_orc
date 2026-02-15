from __future__ import annotations

import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from env_local import load_dotenv
from orchestrator_queue import (
    QueueAgentRunner,
    generate_agentic_message_queued,
    generate_orchestrated_message_queued,
)
from queueing import QueueTaskError
from rag import (
    DEFAULT_MAX_CHARS,
    split_into_chunks,
    truncate_text,
)

DEFAULT_API_KEY = "REPLACE_WITH_OPENAI_KEY"

load_dotenv()

app = FastAPI(title="OpenAI RAG Outreach API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    name: str | None = Field(None, description="Candidate name (optional)")
    resumeText: str = Field(..., description="Full resume as text")
    jdText: str = Field(..., description="Job description text")
    hmNote: str | None = Field(None, description="Hiring manager LinkedIn note")
    role: str | None = Field(None, description="Role title (defaults to Product Manager if omitted)")
    company: str | None = None
    hiringManager: str | None = None
    topK: int = Field(3, ge=1, le=10)
    embeddingModel: str = Field("text-embedding-3-small")
    chatModel: str = Field("gpt-4o-mini")
    outputType: str = Field("message", description="message or cover-letter")
    orchestrate: bool = Field(True, description="Use AI orchestration to choose next action")


class GenerateResponse(BaseModel):
    message: str
    tokenEstimate: int


class DiscoverJobsRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Natural language jobs request")
    page: int = Field(1, ge=1)
    pageSize: int = Field(10, ge=1, le=25)
    country: str = Field("us", description="Country code, currently expected 'us'")
    strict: bool = Field(True, description="Strict verification mode; enforced true in MVP")


class JobListing(BaseModel):
    id: str
    title: str
    company: str
    location: str
    hiringManager: str | None = None
    source: str | None = None
    url: str | None = None
    jdText: str


class DiscoverJobsResponse(BaseModel):
    count: int
    page: int
    pageSize: int
    hasNextPage: bool
    source: str
    jobs: list[JobListing]


class ExtractNameRequest(BaseModel):
    resumeText: str = Field(..., min_length=1, description="Full resume as text")


class ExtractNameResponse(BaseModel):
    name: str | None = None


def _discovery_provider() -> str:
    provider = os.getenv("JOB_DISCOVERY_PROVIDER", "serpapi").strip().lower()
    if provider not in {"serpapi", "seed"}:
        return "serpapi"
    return provider


@app.post("/jobs/discover", response_model=DiscoverJobsResponse)
def discover_jobs_endpoint(request: DiscoverJobsRequest) -> DiscoverJobsResponse:
    load_dotenv()
    provider = _discovery_provider()
    if provider == "serpapi" and not os.getenv("SERPAPI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="SERPAPI_API_KEY is not configured.")
    if provider == "serpapi" and not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    runner = QueueAgentRunner(redis_url=os.getenv("REDIS_URL"))
    try:
        result = runner.discover_jobs(
            provider=provider,
            prompt=request.prompt,
            country=request.country,
            page=request.page,
            page_size=request.pageSize,
            strict=True,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Timed out waiting for discovery worker: {exc}",
        ) from exc
    except QueueTaskError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Discovery worker failure: {exc}",
        ) from exc
    return DiscoverJobsResponse(**result)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    if not api_key or api_key == "REPLACE_WITH_OPENAI_KEY":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    client = OpenAI(api_key=api_key)
    resume_text = truncate_text(request.resumeText.strip(), DEFAULT_MAX_CHARS)
    jd_text = truncate_text(request.jdText.strip(), DEFAULT_MAX_CHARS // 2)
    hm_note = (request.hmNote or "").strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text is empty.")
    if not jd_text:
        raise HTTPException(status_code=400, detail="Job description text is empty.")

    resume_chunks = split_into_chunks(resume_text)
    query_parts: list[str] = []
    if jd_text:
        query_parts.append(f"JOB DESCRIPTION:\n{jd_text}")
    if hm_note:
        # Reduce prompt confusion: this is an inbound note, not the candidate's greeting.
        query_parts.append(f"HIRING MANAGER NOTE (FROM THEM TO CANDIDATE):\n{hm_note}")
    query_text = truncate_text("\n\n".join(query_parts), DEFAULT_MAX_CHARS // 2)
    token_estimate = max(1, (len(resume_text) + len(query_text)) // 4)

    role_value = request.role or "Product Manager"
    candidate_name = request.name or "Candidate"
    runner = QueueAgentRunner(redis_url=os.getenv("REDIS_URL"))

    hm_note_clean = hm_note
    if hm_note_clean:
        # Many inbound LinkedIn notes begin with "Hi <candidate>,"; if we pass that through raw,
        # the model may mistakenly greet the candidate. Strip the leading greeting.
        hm_note_clean = re.sub(
            r"^\s*(hi|hello|hey)\s+[^,\n]{1,40}[,!\n]\s*",
            "",
            hm_note_clean,
            flags=re.IGNORECASE,
        ).strip()
        hm_note = hm_note_clean
    try:
        if request.orchestrate:
            message = generate_orchestrated_message_queued(
                controller_client=client,
                runner=runner,
                candidate=candidate_name,
                role=role_value,
                company=request.company,
                hiring_manager=request.hiringManager,
                query=query_text,
                resume_chunks=resume_chunks,
                embedding_model=request.embeddingModel,
                top_k=request.topK,
                chat_model=request.chatModel,
                output_type=request.outputType,
            )
        else:
            message = generate_agentic_message_queued(
                runner=runner,
                candidate=candidate_name,
                role=role_value,
                company=request.company,
                hiring_manager=request.hiringManager,
                query=query_text,
                resume_chunks=resume_chunks,
                embedding_model=request.embeddingModel,
                top_k=request.topK,
                chat_model=request.chatModel,
                output_type=request.outputType,
            )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Timed out waiting for agent workers: {exc}",
        ) from exc
    except QueueTaskError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent worker failure: {exc}",
        ) from exc
    return GenerateResponse(message=message, tokenEstimate=token_estimate)


def _extract_name_with_llm(client: OpenAI, resume_text: str, *, chat_model: str) -> str | None:
    system = (
        "You extract the candidate's full name from resume text. "
        "Return ONLY strict JSON like {\"name\":\"First Last\"} or {\"name\":null}. "
        "Do not guess; if you cannot find a name in the text, return null."
    )
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"RESUME TEXT:\n{resume_text}"},
        ],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return None
    try:
        import json

        data = json.loads(content)
        name = data.get("name")
        if isinstance(name, str):
            name = name.strip()
            return name or None
        return None
    except Exception:
        return None


@app.post("/resume/extract_name", response_model=ExtractNameResponse)
def extract_name_endpoint(request: ExtractNameRequest) -> ExtractNameResponse:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    if not api_key or api_key == "REPLACE_WITH_OPENAI_KEY":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    resume_text = truncate_text(request.resumeText.strip(), DEFAULT_MAX_CHARS)
    if not resume_text:
        return ExtractNameResponse(name=None)

    model = os.getenv("NAME_EXTRACT_CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    name = _extract_name_with_llm(client, resume_text, chat_model=model)
    return ExtractNameResponse(name=name)
