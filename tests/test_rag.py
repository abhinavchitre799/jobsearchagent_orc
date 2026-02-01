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


def test_api_generate_uses_stubbed_llm(monkeypatch):
    # Patch API dependencies to avoid real OpenAI calls
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = FakeOpenAI()

    def fake_embed_texts(_client, texts, model):
        return [vec.embedding for vec in fake_client.embeddings.create(model=model, input=texts).data]

    monkeypatch.setattr(rag, "embed_texts", fake_embed_texts)
    monkeypatch.setattr("api.generate_agentic_message", lambda *a, **k: "stubbed message")
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
