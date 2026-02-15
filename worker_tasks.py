from __future__ import annotations

import os
from typing import Sequence, Tuple

from openai import OpenAI

from env_local import load_dotenv
from jobs import discover_jobs_seed
from jobs_live import DEFAULT_DISCOVERY_MODEL, discover_jobs_live
from rag import (
    build_llm_message,
    critique_message,
    plan_for_goal,
    retrieve_chunks_with_embeddings,
    revise_message,
)

load_dotenv()


def _openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for worker tasks.")
    return OpenAI(api_key=api_key)


def _to_tuples(retrieved_chunks: Sequence[dict]) -> list[Tuple[str, float]]:
    pairs: list[Tuple[str, float]] = []
    for item in retrieved_chunks:
        chunk = str(item.get("chunk", "")).strip()
        if not chunk:
            continue
        score = float(item.get("score", 0.0))
        pairs.append((chunk, score))
    return pairs


def _to_dicts(retrieved_chunks: Sequence[Tuple[str, float]]) -> list[dict]:
    return [{"chunk": chunk, "score": float(score)} for chunk, score in retrieved_chunks]


def retrieve_agent_task(
    *,
    resume_chunks: Sequence[str],
    query: str,
    embedding_model: str,
    top_k: int,
) -> list[dict]:
    client = _openai_client()
    retrieved = retrieve_chunks_with_embeddings(
        client,
        resume_chunks,
        query,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    return _to_dicts(retrieved)


def planner_agent_task(
    *,
    goal: dict,
    query: str,
    retrieved_chunks: Sequence[dict],
    chat_model: str,
) -> str:
    client = _openai_client()
    return plan_for_goal(
        client,
        goal=goal,
        query=query,
        retrieved_chunks=_to_tuples(retrieved_chunks),
        chat_model=chat_model,
    )


def drafter_agent_task(
    *,
    candidate: str,
    role: str | None,
    company: str | None,
    hiring_manager: str | None,
    query: str,
    retrieved_chunks: Sequence[dict],
    chat_model: str,
    output_type: str,
    goal_text: str | None = None,
    plan_text: str | None = None,
    constraints_text: str | None = None,
) -> str:
    client = _openai_client()
    return build_llm_message(
        client,
        candidate=candidate,
        role=role,
        company=company,
        hiring_manager=hiring_manager,
        query=query,
        retrieved_chunks=_to_tuples(retrieved_chunks),
        chat_model=chat_model,
        output_type=output_type,
        goal_text=goal_text,
        plan_text=plan_text,
        constraints_text=constraints_text,
    )


def critic_agent_task(
    *,
    goal: dict,
    draft: str,
    chat_model: str,
) -> dict:
    client = _openai_client()
    return critique_message(
        client,
        goal=goal,
        draft=draft,
        chat_model=chat_model,
    )


def reviser_agent_task(
    *,
    goal: dict,
    draft: str,
    critique: dict,
    chat_model: str,
) -> str:
    client = _openai_client()
    return revise_message(
        client,
        goal=goal,
        draft=draft,
        critique=critique,
        chat_model=chat_model,
    )


def discover_jobs_agent_task(
    *,
    provider: str,
    prompt: str,
    country: str,
    page: int,
    page_size: int,
    strict: bool,
) -> dict:
    load_dotenv()
    provider_value = (provider or "serpapi").strip().lower()
    if provider_value == "seed":
        return discover_jobs_seed(
            prompt=prompt,
            country=country,
            page=page,
            page_size=page_size,
            strict=strict,
        )
    if provider_value != "serpapi":
        raise RuntimeError(f"Unsupported discovery provider: {provider_value}")

    serpapi_api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not serpapi_api_key:
        raise RuntimeError("SERPAPI_API_KEY is required for SerpAPI job discovery.")

    client = _openai_client()
    model = os.getenv("JOB_DISCOVERY_CHAT_MODEL", DEFAULT_DISCOVERY_MODEL)
    return discover_jobs_live(
        client,
        prompt=prompt,
        country=country,
        page=page,
        page_size=page_size,
        strict=strict,
        serpapi_api_key=serpapi_api_key,
        chat_model=model,
    )
