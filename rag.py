from __future__ import annotations

import hashlib
import json
import textwrap
from typing import Dict, List, Sequence, Tuple

from openai import OpenAI

from prompts import (
    COVER_LETTER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    MESSAGE_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REVISER_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

# Simple in-memory embedding cache: {(model, text_hash): embedding}
_EMBED_CACHE: Dict[tuple[str, str], List[float]] = {}
# Budget guardrails
DEFAULT_MAX_CHARS = 20000  # cap resume/JD inputs to avoid runaway costs
LOW_SCORE_THRESHOLD = 0.08  # filter weak matches by default
MAX_TOP_K = 10
MESSAGE_MAX_TOKENS = 160
COVER_LETTER_MAX_TOKENS = 240
PLAN_MAX_TOKENS = 120
CRITIC_MAX_TOKENS = 120
REVISER_MAX_TOKENS = 160
ORCHESTRATOR_MAX_TOKENS = 120


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip().lower()


def _retrieval_stats(retrieved_chunks: Sequence[Tuple[str, float]]) -> dict:
    if not retrieved_chunks:
        return {"count": 0, "max_score": 0.0, "avg_score": 0.0}
    scores = [score for _, score in retrieved_chunks]
    return {
        "count": len(scores),
        "max_score": max(scores),
        "avg_score": sum(scores) / len(scores),
    }


def _parse_action_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def decide_next_action(
    client: OpenAI,
    *,
    state: dict,
    chat_model: str,
) -> dict:
    """Use an LLM to choose the next orchestration action."""
    prompt = textwrap.dedent(
        f"""
        Current state (JSON):
        {json.dumps(state, indent=2)}

        Choose the next action.
        """
    ).strip()
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=ORCHESTRATOR_MAX_TOKENS,
    )
    raw = completion.choices[0].message.content.strip()
    return _parse_action_json(raw)


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


def summarize_chunk(chunk: str, max_len: int = 140) -> str:
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
    goal_text: str | None = None,
    plan_text: str | None = None,
    constraints_text: str | None = None,
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
        goal_text=goal_text or "Not provided",
        plan_text=plan_text or "Not provided",
        constraints_text=constraints_text or "Not provided",
        output_type=output_type.replace("-", " "),
    )
    max_tokens = COVER_LETTER_MAX_TOKENS if output_type == "cover-letter" else MESSAGE_MAX_TOKENS
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
        ],
        temperature=0.4,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def build_goal(
    *,
    candidate: str,
    role: str | None,
    company: str | None,
    output_type: str,
) -> dict:
    role_value = role or "Not specified"
    company_value = company or "Not specified"
    objective = f"Write a {output_type.replace('-', ' ')} for {candidate} targeting {role_value} at {company_value}."
    if output_type == "cover-letter":
        length_cap = "220 words"
    else:
        length_cap = "120 words"
    success_criteria = [
        "Uses only resume-backed facts",
        "Includes 2-3 resume-aligned highlights with '(from resume)'",
        "States interest in the role/company",
        "Closes with a light call to action",
        f"Stays under {length_cap}",
    ]
    constraints = [
        "No hallucinated facts",
        "Warm, professional, concise tone",
        "Use hiring manager name only if provided",
    ]
    return {
        "objective": objective,
        "success_criteria": success_criteria,
        "constraints": constraints,
    }


def plan_for_goal(
    client: OpenAI,
    *,
    goal: dict,
    query: str,
    retrieved_chunks: Sequence[Tuple[str, float]],
    chat_model: str,
) -> str:
    snippet_text = "\n".join(f"- {summarize_chunk(chunk)}" for chunk, _ in retrieved_chunks)
    prompt = textwrap.dedent(
        f"""
        Goal: {goal.get('objective')}
        Success criteria: {', '.join(goal.get('success_criteria', []))}
        Constraints: {', '.join(goal.get('constraints', []))}

        Job context:
        {query}

        Resume snippets:
        {snippet_text or '- None provided'}
        """
    ).strip()
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=PLAN_MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


def critique_message(
    client: OpenAI,
    *,
    goal: dict,
    draft: str,
    chat_model: str,
) -> dict:
    prompt = textwrap.dedent(
        f"""
        Goal: {goal.get('objective')}
        Success criteria: {', '.join(goal.get('success_criteria', []))}
        Constraints: {', '.join(goal.get('constraints', []))}

        Draft:
        {draft}
        """
    ).strip()
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=CRITIC_MAX_TOKENS,
    )
    raw = completion.choices[0].message.content.strip()
    try:
        critique = json.loads(raw)
    except json.JSONDecodeError:
        return {"pass": True, "issues": [], "suggestions": []}
    if not isinstance(critique, dict):
        return {"pass": True, "issues": [], "suggestions": []}
    critique.setdefault("pass", True)
    critique.setdefault("issues", [])
    critique.setdefault("suggestions", [])
    return critique


def revise_message(
    client: OpenAI,
    *,
    goal: dict,
    draft: str,
    critique: dict,
    chat_model: str,
) -> str:
    prompt = textwrap.dedent(
        f"""
        Goal: {goal.get('objective')}
        Constraints: {', '.join(goal.get('constraints', []))}

        Draft:
        {draft}

        Issues:
        {', '.join(critique.get('issues', [])) or 'None'}

        Suggestions:
        {', '.join(critique.get('suggestions', [])) or 'None'}
        """
    ).strip()
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": REVISER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=REVISER_MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


def generate_agentic_message(
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
    goal = build_goal(
        candidate=candidate,
        role=role,
        company=company,
        output_type=output_type,
    )
    plan = plan_for_goal(
        client,
        goal=goal,
        query=query,
        retrieved_chunks=retrieved_chunks,
        chat_model=chat_model,
    )
    constraints_text = "; ".join(goal.get("constraints", []))
    success_text = "; ".join(goal.get("success_criteria", []))
    goal_text = f"{goal.get('objective')} Success criteria: {success_text}"
    draft = build_llm_message(
        client,
        candidate=candidate,
        role=role,
        company=company,
        hiring_manager=hiring_manager,
        query=query,
        retrieved_chunks=retrieved_chunks,
        chat_model=chat_model,
        output_type=output_type,
        goal_text=goal_text,
        plan_text=plan,
        constraints_text=constraints_text,
    )
    critique = critique_message(
        client,
        goal=goal,
        draft=draft,
        chat_model=chat_model,
    )
    if not critique.get("pass", True):
        draft = revise_message(
            client,
            goal=goal,
            draft=draft,
            critique=critique,
            chat_model=chat_model,
        )
    return draft


def generate_orchestrated_message(
    client: OpenAI,
    *,
    candidate: str,
    role: str | None,
    company: str | None,
    hiring_manager: str | None,
    query: str,
    resume_chunks: Sequence[str],
    embedding_model: str,
    top_k: int,
    chat_model: str,
    output_type: str = "message",
) -> str:
    max_revisions = 2
    revision_count = 0
    goal = build_goal(
        candidate=candidate,
        role=role,
        company=company,
        output_type=output_type,
    )
    goal_text = f"{goal.get('objective')} Success criteria: {'; '.join(goal.get('success_criteria', []))}"
    constraints_text = "; ".join(goal.get("constraints", []))

    current_top_k = max(1, min(top_k, MAX_TOP_K))
    retrieved = retrieve_chunks_with_embeddings(
        client,
        resume_chunks,
        query,
        embedding_model=embedding_model,
        top_k=current_top_k,
    )

    plan_text: str | None = None
    draft: str | None = None
    critique: dict | None = None
    last_action: str | None = None
    last_signature: str | None = None
    stagnant_cycles = 0

    while True:
        if critique and critique.get("pass", False):
            return draft or ""

        stats = _retrieval_stats(retrieved)
        state = {
            "has_plan": bool(plan_text),
            "has_draft": bool(draft),
            "last_action": last_action,
            "critique_pass": critique.get("pass") if critique else None,
            "critique_issues": critique.get("issues", []) if critique else [],
            "revisions_done": revision_count,
            "revisions_remaining": max(0, max_revisions - revision_count),
            "retrieval": {
                "top_k": current_top_k,
                "count": stats["count"],
                "max_score": round(stats["max_score"], 4),
                "avg_score": round(stats["avg_score"], 4),
            },
            "output_type": output_type,
        }
        signature = json.dumps(
            {
                "plan": _normalize_text(plan_text),
                "draft": _normalize_text(draft),
                "critique": critique,
                "retrieval": state["retrieval"],
                "last_action": last_action,
            },
            sort_keys=True,
        )
        if signature == last_signature:
            stagnant_cycles += 1
        else:
            stagnant_cycles = 0
            last_signature = signature
        if stagnant_cycles >= 2:
            if draft:
                return draft
            # Fallback to a single deterministic draft if we got stuck early.
            draft = build_llm_message(
                client,
                candidate=candidate,
                role=role,
                company=company,
                hiring_manager=hiring_manager,
                query=query,
                retrieved_chunks=retrieved,
                chat_model=chat_model,
                output_type=output_type,
                goal_text=goal_text,
                plan_text=plan_text,
                constraints_text=constraints_text,
            )
            return draft

        decision = decide_next_action(client, state=state, chat_model=chat_model)
        action = decision.get("action")
        if action not in {
            "retrieve_more",
            "plan",
            "draft",
            "critique",
            "revise",
            "finalize",
        }:
            action = None

        if action == "finalize" and draft:
            return draft

        if action == "retrieve_more":
            requested_top_k = decision.get("next_top_k")
            if isinstance(requested_top_k, int):
                requested_top_k = max(1, min(requested_top_k, MAX_TOP_K))
            else:
                requested_top_k = min(current_top_k + 2, MAX_TOP_K)
            if requested_top_k == current_top_k:
                action = None
            else:
                current_top_k = requested_top_k
                retrieved = retrieve_chunks_with_embeddings(
                    client,
                    resume_chunks,
                    query,
                    embedding_model=embedding_model,
                    top_k=current_top_k,
                )
                last_action = "retrieve_more"
                continue

        if action is None:
            # Safe fallback to deterministic next step.
            if not plan_text:
                action = "plan"
            elif not draft:
                action = "draft"
            elif critique is None:
                action = "critique"
            elif critique.get("pass", False):
                return draft or ""
            else:
                action = "revise"

        if action == "plan":
            plan_text = plan_for_goal(
                client,
                goal=goal,
                query=query,
                retrieved_chunks=retrieved,
                chat_model=chat_model,
            )
            last_action = "plan"
            continue

        if action == "critique":
            if not draft:
                action = "draft"
            else:
                critique = critique_message(
                    client,
                    goal=goal,
                    draft=draft,
                    chat_model=chat_model,
                )
                last_action = "critique"
                continue
        if action == "revise":
            if revision_count >= max_revisions:
                return draft or ""
            if not draft:
                action = "draft"
            else:
                previous = draft
                critique = critique or {"pass": False, "issues": [], "suggestions": []}
                draft = revise_message(
                    client,
                    goal=goal,
                    draft=draft,
                    critique=critique,
                    chat_model=chat_model,
                )
                critique = None
                revision_count += 1
                if _normalize_text(previous) == _normalize_text(draft):
                    return draft
                last_action = "revise"
                continue
        if action == "draft":
            previous = draft
            draft = build_llm_message(
                client,
                candidate=candidate,
                role=role,
                company=company,
                hiring_manager=hiring_manager,
                query=query,
                retrieved_chunks=retrieved,
                chat_model=chat_model,
                output_type=output_type,
                goal_text=goal_text,
                plan_text=plan_text,
                constraints_text=constraints_text,
            )
            if previous and _normalize_text(previous) == _normalize_text(draft):
                return draft
            last_action = "draft"
            continue
