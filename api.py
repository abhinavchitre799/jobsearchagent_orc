from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from rag import (
    DEFAULT_MAX_CHARS,
    generate_agentic_message,
    generate_orchestrated_message,
    retrieve_chunks_with_embeddings,
    split_into_chunks,
    truncate_text,
)

DEFAULT_API_KEY = "REPLACE_WITH_OPENAI_KEY"

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


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
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
    query_text = truncate_text(
        "\n\n".join([part for part in [jd_text, hm_note] if part]),
        DEFAULT_MAX_CHARS // 2,
    )
    token_estimate = max(1, (len(resume_text) + len(query_text)) // 4)

    role_value = request.role or "Product Manager"
    candidate_name = request.name or "Candidate"
    if request.orchestrate:
        message = generate_orchestrated_message(
            client,
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
        retrieved = retrieve_chunks_with_embeddings(
            client,
            resume_chunks,
            query_text,
            embedding_model=request.embeddingModel,
            top_k=request.topK,
        )
        message = generate_agentic_message(
            client,
            candidate=candidate_name,
            role=role_value,
            company=request.company,
            hiring_manager=request.hiringManager,
            query=query_text,
            retrieved_chunks=retrieved,
            chat_model=request.chatModel,
            output_type=request.outputType,
        )
    return GenerateResponse(message=message, tokenEstimate=token_estimate)
