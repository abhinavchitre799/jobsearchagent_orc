from __future__ import annotations

import hashlib
import textwrap
from typing import Dict, List, Sequence, Tuple

from openai import OpenAI

from prompts import (
    COVER_LETTER_SYSTEM_PROMPT,
    MESSAGE_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

# Simple in-memory embedding cache: {(model, text_hash): embedding}
_EMBED_CACHE: Dict[tuple[str, str], List[float]] = {}
# Budget guardrails
DEFAULT_MAX_CHARS = 20000  # cap resume/JD inputs to avoid runaway costs
LOW_SCORE_THRESHOLD = 0.08  # filter weak matches by default


def split_into_chunks(text: str, max_chars: int = 900) -> List[str]:
    """Split resume into readable chunks, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        current = para
        while len(current) > max_chars:
            split_at = current.rfind(". ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            chunks.append(current[: split_at + 1].strip())
            current = current[split_at + 1 :].strip()
        if current:
            chunks.append(current)
    if not chunks:
        chunks.append(text.strip())
    return chunks


def summarize_chunk(chunk: str, max_len: int = 180) -> str:
    """Keep the chunk concise for bullet points."""
    clean = " ".join(chunk.split())
    if len(clean) <= max_len:
        return clean
    cutoff = clean.rfind(" ", 0, max_len)
    if cutoff == -1:
        cutoff = max_len
    return clean[:cutoff].rstrip() + "..."


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    if not numerator:
        return 0.0
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return numerator / (norm_a * norm_b)


def truncate_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Trim text to a maximum character budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _cache_key(model: str, text: str) -> tuple[str, str]:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (model, digest)


def embed_texts(client: OpenAI, texts: Sequence[str], model: str) -> List[List[float]]:
    """Request embeddings for a list of texts, cached per (model, text hash)."""
    to_request: List[str] = []
    results: List[List[float]] = []
    keys = [_cache_key(model, t) for t in texts]
    for key, text in zip(keys, texts):
        if key in _EMBED_CACHE:
            results.append(_EMBED_CACHE[key])
        else:
            to_request.append(text)
            results.append(None)  # placeholder

    if to_request:
        response = client.embeddings.create(model=model, input=list(to_request))
        new_embeddings = [item.embedding for item in response.data]
        # Fill placeholders and cache
        new_iter = iter(new_embeddings)
        for idx, (key, current) in enumerate(zip(keys, results)):
            if current is None:
                emb = next(new_iter)
                _EMBED_CACHE[key] = emb
                results[idx] = emb

    return results


def retrieve_chunks_with_embeddings(
    client: OpenAI,
    resume_chunks: Sequence[str],
    query: str,
    *,
    embedding_model: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    """Return top-k resume chunks scored by cosine similarity against query embedding."""
    chunk_embeddings = embed_texts(client, resume_chunks, embedding_model)
    query_embedding = embed_texts(client, [query], embedding_model)[0]
    scored = []
    for chunk, emb in zip(resume_chunks, chunk_embeddings):
        scored.append((chunk, cosine_similarity(emb, query_embedding)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    filtered = [(c, s) for c, s in scored[: top_k * 2] if s >= LOW_SCORE_THRESHOLD]
    return filtered[:top_k] if filtered else scored[:top_k]


def build_llm_message(
    client: OpenAI,
    *,
    candidate: str,
    role: str | None,
    company: str | None,
    hiring_manager: str | None,
    query: str,
    retrieved_chunks: Sequence[Tuple[str, float]],
    chat_model: str,
    output_type: str = "message",
) -> str:
    snippet_text = "\n".join(f"- {summarize_chunk(chunk)}" for chunk, _ in retrieved_chunks)
    system_prompt = (
        COVER_LETTER_SYSTEM_PROMPT
        if output_type == "cover-letter"
        else MESSAGE_SYSTEM_PROMPT
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        candidate=candidate,
        role=role or "Not specified",
        company=company or "Not specified",
        hiring_manager=hiring_manager or "Not specified",
        query=query,
        snippet_text=snippet_text or "- None provided",
        output_type=output_type.replace("-", " "),
    )
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
        ],
        temperature=0.4,
        max_tokens=260,
    )
    return completion.choices[0].message.content.strip()
