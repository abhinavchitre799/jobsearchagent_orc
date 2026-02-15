import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag
from api import app
from queueing import QueueTaskError
from rag import (
    build_llm_message,
    retrieve_chunks_with_embeddings,
    split_into_chunks,
)


class FakeOpenAI:
    """Minimal fake OpenAI client for offline tests."""

    class _Embeddings:
        def create(self, model, input):
            # Embed by counting keyword occurrences to keep cosine deterministic.
            data = []
            for text in input:
                tokens = text.lower().split()
                vec = [
                    tokens.count("data"),
                    tokens.count("product"),
                    len(tokens),
                ]
                data.append(SimpleNamespace(embedding=vec))
            return SimpleNamespace(data=data)

    class _ChatCompletions:
        def __init__(self, response_text: str):
            self.response_text = response_text

        def create(self, **kwargs):
            message = SimpleNamespace(content=self.response_text)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def __init__(self, response_text: str = "stubbed message"):
        self.embeddings = self._Embeddings()
        self.chat = SimpleNamespace(completions=self._ChatCompletions(response_text))


def test_split_into_chunks_respects_max():
    text = "Para one. " * 200  # long paragraph
    chunks = split_into_chunks(text, max_chars=100)
    assert all(len(c) <= 120 for c in chunks)  # includes sentence split fudge
    assert len(chunks) > 1


def test_cosine_and_retrieval_prefers_relevant_chunk(monkeypatch):
    client = FakeOpenAI()
    # monkeypatch rag.embed_texts to use fake embeddings with matching signature
    def fake_embed_texts(_client, texts, model):
        return [vec.embedding for vec in client.embeddings.create(model=model, input=texts).data]

    monkeypatch.setattr(rag, "embed_texts", fake_embed_texts)
    resume_chunks = [
        "Built data pipelines and analytics.",
        "Managed product roadmap.",
        "Unrelated content.",
    ]
    scored = retrieve_chunks_with_embeddings(
        client, resume_chunks, "data analytics role", embedding_model="fake", top_k=2
    )
    top_chunk, score = scored[0]
    assert "data pipelines" in top_chunk
    assert score > 0


def test_build_llm_message_uses_chat(monkeypatch):
    client = FakeOpenAI(response_text="Hello there")
    msg = build_llm_message(
        client,
        candidate="Alex",
        role="Engineer",
        company="Acme",
        hiring_manager="Jordan",
        query="We need engineers",
        retrieved_chunks=[("Built systems", 0.9)],
        chat_model="fake-model",
    )
    assert "Hello there" in msg


def test_cover_letter_retries_when_truncated():
    class RetryClient:
        class _ChatCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    msg = SimpleNamespace(content="partial cover letter")
                    choice = SimpleNamespace(message=msg, finish_reason="length")
                    return SimpleNamespace(choices=[choice])
                msg = SimpleNamespace(content="complete cover letter")
                choice = SimpleNamespace(message=msg, finish_reason="stop")
                return SimpleNamespace(choices=[choice])

        def __init__(self):
            self.chat = SimpleNamespace(completions=self._ChatCompletions())

    client = RetryClient()
    msg = build_llm_message(
        client,
        candidate="Alex",
        role="Product Manager",
        company="Acme",
        hiring_manager="Jordan",
        query="We need PMs",
        retrieved_chunks=[("Built systems", 0.9)],
        chat_model="fake-model",
        output_type="cover-letter",
    )
    assert msg == "complete cover letter"
    assert client.chat.completions.calls == 2


def test_api_generate_uses_stubbed_llm(monkeypatch):
    # Patch API dependencies to avoid real OpenAI calls
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = FakeOpenAI()

    class DummyRunner:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("api.QueueAgentRunner", DummyRunner)
    monkeypatch.setattr("api.generate_agentic_message_queued", lambda *a, **k: "stubbed message")
    monkeypatch.setattr("api.OpenAI", lambda api_key=None: fake_client)
    client = TestClient(app)
    payload = {
        "name": "Alex",
        "resumeText": "Built data systems and APIs.",
        "jdText": "Looking for a data engineer to build pipelines.",
        "hmNote": "Ping me if interested",
        "orchestrate": False,
    }
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "stubbed message"
    assert data["tokenEstimate"] > 0


def test_api_extract_name_prefills_from_resume(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = FakeOpenAI(response_text='{"name":"Alex Rivera"}')

    monkeypatch.setattr("api.OpenAI", lambda api_key=None: fake_client)
    client = TestClient(app)
    payload = {"resumeText": "Alex Rivera\nProduct Manager\nExperience: ... " + ("x" * 300)}
    resp = client.post("/resume/extract_name", json=payload)
    assert resp.status_code == 200
    assert resp.json()["name"] == "Alex Rivera"


def test_api_extract_name_returns_null_when_unknown(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = FakeOpenAI(response_text='{"name":null}')

    monkeypatch.setattr("api.OpenAI", lambda api_key=None: fake_client)
    client = TestClient(app)
    payload = {"resumeText": "Experience: built things." + ("x" * 300)}
    resp = client.post("/resume/extract_name", json=payload)
    assert resp.status_code == 200
    assert resp.json()["name"] is None


def test_api_extract_name_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    payload = {"resumeText": "Alex Rivera" + ("x" * 300)}
    resp = client.post("/resume/extract_name", json=payload)
    assert resp.status_code == 500


def test_api_discover_jobs_returns_paginated_schema(monkeypatch):
    monkeypatch.setenv("JOB_DISCOVERY_PROVIDER", "seed")

    class DummyRunner:
        def __init__(self, *args, **kwargs):
            pass

        def discover_jobs(self, **kwargs):
            return {
                "count": 2,
                "page": 1,
                "pageSize": 10,
                "hasNextPage": False,
                "source": "seed",
                "jobs": [
                    {
                        "id": "job-1",
                        "title": "Product Manager",
                        "company": "Acme",
                        "location": "Remote (US)",
                        "hiringManager": "Jordan Lee",
                        "source": "seed",
                        "url": "https://example.com/job-1",
                        "jdText": "Own roadmap and analytics outcomes.",
                    }
                ],
            }

    monkeypatch.setattr("api.QueueAgentRunner", DummyRunner)
    client = TestClient(app)
    payload = {
        "prompt": "Fetch all PM jobs in US",
        "page": 1,
        "pageSize": 10,
        "country": "us",
        "strict": True,
    }
    resp = client.post("/jobs/discover", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["page"] == 1
    assert data["pageSize"] == 10
    assert data["hasNextPage"] is False
    assert data["source"] == "seed"
    assert len(data["jobs"]) == 1
    first = data["jobs"][0]
    assert first["title"]
    assert first["company"]
    assert first["jdText"]


def test_api_discover_jobs_requires_serpapi_key(monkeypatch):
    monkeypatch.setenv("JOB_DISCOVERY_PROVIDER", "serpapi")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    client = TestClient(app)
    payload = {"prompt": "Fetch all PM jobs in US"}
    resp = client.post("/jobs/discover", json=payload)
    assert resp.status_code == 503


def test_api_discover_maps_worker_failure(monkeypatch):
    monkeypatch.setenv("JOB_DISCOVERY_PROVIDER", "seed")

    class FailingRunner:
        def __init__(self, *args, **kwargs):
            pass

        def discover_jobs(self, **kwargs):
            raise QueueTaskError("boom")

    monkeypatch.setattr("api.QueueAgentRunner", FailingRunner)
    client = TestClient(app)
    payload = {"prompt": "Fetch all PM jobs in US"}
    resp = client.post("/jobs/discover", json=payload)
    assert resp.status_code == 502


def test_api_discover_maps_timeout(monkeypatch):
    monkeypatch.setenv("JOB_DISCOVERY_PROVIDER", "seed")

    class TimeoutRunner:
        def __init__(self, *args, **kwargs):
            pass

        def discover_jobs(self, **kwargs):
            raise TimeoutError("slow")

    monkeypatch.setattr("api.QueueAgentRunner", TimeoutRunner)
    client = TestClient(app)
    payload = {"prompt": "Fetch all PM jobs in US"}
    resp = client.post("/jobs/discover", json=payload)
    assert resp.status_code == 504


def test_orchestrated_stops_on_no_improvement(monkeypatch):
    client = FakeOpenAI(response_text="same draft")

    monkeypatch.setattr(
        rag,
        "retrieve_chunks_with_embeddings",
        lambda *a, **k: [("chunk", 0.2)],
    )

    draft_calls = {"count": 0}

    def fake_build(*_args, **_kwargs):
        draft_calls["count"] += 1
        return "same draft"

    monkeypatch.setattr(rag, "build_llm_message", fake_build)
    monkeypatch.setattr(rag, "decide_next_action", lambda *a, **k: {"action": "draft"})

    message = rag.generate_orchestrated_message(
        client,
        candidate="Alex",
        role="Engineer",
        company="Acme",
        hiring_manager="Jordan",
        query="We need engineers",
        resume_chunks=["Built systems."],
        embedding_model="fake",
        top_k=3,
        chat_model="fake-model",
        output_type="message",
    )

    assert message == "same draft"
    assert draft_calls["count"] == 2


def test_orchestrated_caps_revisions(monkeypatch):
    client = FakeOpenAI(response_text="irrelevant")

    monkeypatch.setattr(
        rag,
        "retrieve_chunks_with_embeddings",
        lambda *a, **k: [("chunk", 0.2)],
    )
    monkeypatch.setattr(rag, "decide_next_action", lambda *a, **k: {"action": "revise"})

    draft_calls = {"count": 0}
    revise_calls = {"count": 0}

    def fake_build(*_args, **_kwargs):
        draft_calls["count"] += 1
        return "draft-0"

    def fake_revise(*_args, **_kwargs):
        revise_calls["count"] += 1
        return f"draft-{revise_calls['count']}"

    monkeypatch.setattr(rag, "build_llm_message", fake_build)
    monkeypatch.setattr(rag, "revise_message", fake_revise)

    message = rag.generate_orchestrated_message(
        client,
        candidate="Alex",
        role="Engineer",
        company="Acme",
        hiring_manager="Jordan",
        query="We need engineers",
        resume_chunks=["Built systems."],
        embedding_model="fake",
        top_k=3,
        chat_model="fake-model",
        output_type="message",
    )

    assert message == "draft-2"
    assert draft_calls["count"] == 1
    assert revise_calls["count"] == 2
