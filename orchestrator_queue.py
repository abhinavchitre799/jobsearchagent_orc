from __future__ import annotations

import json
from typing import Sequence, Tuple

from openai import OpenAI

from queueing import enqueue_task, get_queue, get_redis_connection, wait_for_job
from rag import MAX_TOP_K, build_goal, decide_next_action

RETRIEVER_QUEUE = "retriever"
PLANNER_QUEUE = "planner"
DRAFTER_QUEUE = "drafter"
CRITIC_QUEUE = "critic"
REVISER_QUEUE = "reviser"
JOB_DISCOVERY_QUEUE = "job_discovery"


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


def _to_dicts(retrieved_chunks: Sequence[Tuple[str, float]]) -> list[dict]:
    return [{"chunk": chunk, "score": float(score)} for chunk, score in retrieved_chunks]


def _to_tuples(retrieved_chunks: Sequence[dict]) -> list[Tuple[str, float]]:
    pairs: list[Tuple[str, float]] = []
    for item in retrieved_chunks:
        chunk = str(item.get("chunk", "")).strip()
        if not chunk:
            continue
        score = float(item.get("score", 0.0))
        pairs.append((chunk, score))
    return pairs


class QueueAgentRunner:
    def __init__(self, *, redis_url: str | None = None):
        self.connection = get_redis_connection(redis_url)
        self.queues = {
            RETRIEVER_QUEUE: get_queue(RETRIEVER_QUEUE, connection=self.connection),
            PLANNER_QUEUE: get_queue(PLANNER_QUEUE, connection=self.connection),
            DRAFTER_QUEUE: get_queue(DRAFTER_QUEUE, connection=self.connection),
            CRITIC_QUEUE: get_queue(CRITIC_QUEUE, connection=self.connection),
            REVISER_QUEUE: get_queue(REVISER_QUEUE, connection=self.connection),
            JOB_DISCOVERY_QUEUE: get_queue(JOB_DISCOVERY_QUEUE, connection=self.connection),
        }

    def _run(self, *, queue_name: str, task_path: str, kwargs: dict, description: str):
        job = enqueue_task(
            self.queues[queue_name],
            task_path,
            kwargs=kwargs,
            description=description,
        )
        return wait_for_job(job=job)

    def retrieve(
        self,
        *,
        resume_chunks: Sequence[str],
        query: str,
        embedding_model: str,
        top_k: int,
    ) -> list[Tuple[str, float]]:
        result = self._run(
            queue_name=RETRIEVER_QUEUE,
            task_path="worker_tasks.retrieve_agent_task",
            kwargs={
                "resume_chunks": list(resume_chunks),
                "query": query,
                "embedding_model": embedding_model,
                "top_k": top_k,
            },
            description=f"retrieve top_k={top_k}",
        )
        return _to_tuples(result)

    def plan(
        self,
        *,
        goal: dict,
        query: str,
        retrieved_chunks: Sequence[Tuple[str, float]],
        chat_model: str,
    ) -> str:
        return self._run(
            queue_name=PLANNER_QUEUE,
            task_path="worker_tasks.planner_agent_task",
            kwargs={
                "goal": goal,
                "query": query,
                "retrieved_chunks": _to_dicts(retrieved_chunks),
                "chat_model": chat_model,
            },
            description="planner step",
        )

    def draft(
        self,
        *,
        candidate: str,
        role: str | None,
        company: str | None,
        hiring_manager: str | None,
        query: str,
        retrieved_chunks: Sequence[Tuple[str, float]],
        chat_model: str,
        output_type: str,
        goal_text: str | None = None,
        plan_text: str | None = None,
        constraints_text: str | None = None,
    ) -> str:
        return self._run(
            queue_name=DRAFTER_QUEUE,
            task_path="worker_tasks.drafter_agent_task",
            kwargs={
                "candidate": candidate,
                "role": role,
                "company": company,
                "hiring_manager": hiring_manager,
                "query": query,
                "retrieved_chunks": _to_dicts(retrieved_chunks),
                "chat_model": chat_model,
                "output_type": output_type,
                "goal_text": goal_text,
                "plan_text": plan_text,
                "constraints_text": constraints_text,
            },
            description=f"draft {output_type}",
        )

    def critique(
        self,
        *,
        goal: dict,
        draft: str,
        chat_model: str,
    ) -> dict:
        return self._run(
            queue_name=CRITIC_QUEUE,
            task_path="worker_tasks.critic_agent_task",
            kwargs={
                "goal": goal,
                "draft": draft,
                "chat_model": chat_model,
            },
            description="critique step",
        )

    def revise(
        self,
        *,
        goal: dict,
        draft: str,
        critique: dict,
        chat_model: str,
    ) -> str:
        return self._run(
            queue_name=REVISER_QUEUE,
            task_path="worker_tasks.reviser_agent_task",
            kwargs={
                "goal": goal,
                "draft": draft,
                "critique": critique,
                "chat_model": chat_model,
            },
            description="revise step",
        )

    def discover_jobs(
        self,
        *,
        provider: str,
        prompt: str,
        country: str,
        page: int,
        page_size: int,
        strict: bool,
    ) -> dict:
        return self._run(
            queue_name=JOB_DISCOVERY_QUEUE,
            task_path="worker_tasks.discover_jobs_agent_task",
            kwargs={
                "provider": provider,
                "prompt": prompt,
                "country": country,
                "page": page,
                "page_size": page_size,
                "strict": strict,
            },
            description=f"discover jobs provider={provider} page={page}",
        )


def generate_agentic_message_queued(
    *,
    runner: QueueAgentRunner,
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
    goal = build_goal(
        candidate=candidate,
        role=role,
        company=company,
        output_type=output_type,
    )
    retrieved = runner.retrieve(
        resume_chunks=resume_chunks,
        query=query,
        embedding_model=embedding_model,
        top_k=max(1, min(top_k, MAX_TOP_K)),
    )
    plan = runner.plan(
        goal=goal,
        query=query,
        retrieved_chunks=retrieved,
        chat_model=chat_model,
    )
    constraints_text = "; ".join(goal.get("constraints", []))
    success_text = "; ".join(goal.get("success_criteria", []))
    goal_text = f"{goal.get('objective')} Success criteria: {success_text}"
    draft = runner.draft(
        candidate=candidate,
        role=role,
        company=company,
        hiring_manager=hiring_manager,
        query=query,
        retrieved_chunks=retrieved,
        chat_model=chat_model,
        output_type=output_type,
        goal_text=goal_text,
        plan_text=plan,
        constraints_text=constraints_text,
    )
    critique = runner.critique(
        goal=goal,
        draft=draft,
        chat_model=chat_model,
    )
    if not critique.get("pass", True):
        draft = runner.revise(
            goal=goal,
            draft=draft,
            critique=critique,
            chat_model=chat_model,
        )
    return draft


def generate_orchestrated_message_queued(
    *,
    controller_client: OpenAI,
    runner: QueueAgentRunner,
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
    retrieved = runner.retrieve(
        resume_chunks=resume_chunks,
        query=query,
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
            draft = runner.draft(
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

        decision = decide_next_action(controller_client, state=state, chat_model=chat_model)
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
                retrieved = runner.retrieve(
                    resume_chunks=resume_chunks,
                    query=query,
                    embedding_model=embedding_model,
                    top_k=current_top_k,
                )
                last_action = "retrieve_more"
                continue

        if action is None:
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
            plan_text = runner.plan(
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
                critique = runner.critique(
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
                draft = runner.revise(
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
            draft = runner.draft(
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
