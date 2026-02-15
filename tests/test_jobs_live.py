from __future__ import annotations

from types import SimpleNamespace

import jobs_live


class FakeIntentClient:
    class _ChatCompletions:
        def __init__(self, content: str):
            self.content = content

        def create(self, **kwargs):
            msg = SimpleNamespace(content=self.content)
            choice = SimpleNamespace(message=msg)
            return SimpleNamespace(choices=[choice])

    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=self._ChatCompletions(content))


def test_parse_prompt_to_intent_success():
    client = FakeIntentClient(
        '{"role_keywords":["product manager"],"location":"United States","seniority":"senior","must_have_terms":["analytics"],"exclude_terms":["intern"],"country":"us"}'
    )
    intent = jobs_live.parse_prompt_to_intent(
        client,
        prompt="Fetch all PM jobs in US",
        country="us",
        chat_model="fake",
    )
    assert intent["role_keywords"] == ["product manager"]
    assert intent["location"] == "United States"
    assert intent["seniority"] == "senior"
    assert intent["must_have_terms"] == ["analytics"]
    assert intent["exclude_terms"] == ["intern"]
    assert intent["country"] == "us"


def test_parse_prompt_to_intent_malformed_fallback():
    client = FakeIntentClient("not-json")
    intent = jobs_live.parse_prompt_to_intent(
        client,
        prompt="fetch PM jobs",
        country="us",
        chat_model="fake",
    )
    assert "product manager" in intent["role_keywords"]
    assert intent["country"] == "us"


def test_strict_verification_drops_invalid_jobs():
    result = jobs_live.strict_verify_rank_paginate(
        jobs=[
            {
                "id": "",
                "title": "Product Manager",
                "company": "Acme",
                "location": "Remote",
                "hiringManager": None,
                "source": "serpapi",
                "url": "https://example.com/job-1",
                "jdText": "Own roadmap",
            },
            {
                "id": "",
                "title": "",
                "company": "Acme",
                "location": "Remote",
                "hiringManager": None,
                "source": "serpapi",
                "url": "https://example.com/job-2",
                "jdText": "Invalid missing title",
            },
        ],
        prompt="Fetch PM jobs",
        intent=jobs_live._default_intent("Fetch PM jobs", "us"),
        strict=True,
        page=1,
        page_size=10,
    )
    assert result["count"] == 1
    assert result["jobs"][0]["title"] == "Product Manager"


def test_dedupe_by_url_and_id():
    base = {
        "title": "Product Manager",
        "company": "Acme",
        "location": "Remote",
        "hiringManager": None,
        "source": "serpapi",
        "jdText": "Roadmap ownership",
    }
    result = jobs_live.strict_verify_rank_paginate(
        jobs=[
            {**base, "id": "same-id", "url": "https://example.com/job-1"},
            {**base, "id": "same-id", "url": "https://example.com/job-2"},
            {**base, "id": "other-id", "url": "https://example.com/job-1"},
        ],
        prompt="Fetch PM jobs",
        intent=jobs_live._default_intent("Fetch PM jobs", "us"),
        strict=True,
        page=1,
        page_size=10,
    )
    assert result["count"] == 1


def test_ranking_prefers_title_match():
    jobs = [
        {
            "id": "",
            "title": "Program Manager",
            "company": "Acme",
            "location": "US",
            "hiringManager": None,
            "source": "serpapi",
            "url": "https://example.com/program",
            "jdText": "General operations",
        },
        {
            "id": "",
            "title": "Senior Product Manager",
            "company": "Acme",
            "location": "US",
            "hiringManager": None,
            "source": "serpapi",
            "url": "https://example.com/product",
            "jdText": "Roadmap and analytics",
        },
    ]
    intent = {
        "role_keywords": ["product manager"],
        "location": "United States",
        "seniority": "",
        "must_have_terms": [],
        "exclude_terms": [],
        "country": "us",
    }
    result = jobs_live.strict_verify_rank_paginate(
        jobs=jobs,
        prompt="Fetch all PM jobs in US",
        intent=intent,
        strict=True,
        page=1,
        page_size=10,
    )
    assert result["count"] == 2
    assert result["jobs"][0]["title"] == "Senior Product Manager"


def test_pagination_for_top_results():
    jobs = []
    for idx in range(25):
        jobs.append(
            {
                "id": f"job-{idx}",
                "title": "Product Manager",
                "company": f"Company {idx}",
                "location": "US",
                "hiringManager": None,
                "source": "serpapi",
                "url": f"https://example.com/job-{idx}",
                "jdText": "Product strategy",
            }
        )
    result = jobs_live.strict_verify_rank_paginate(
        jobs=jobs,
        prompt="Fetch all PM jobs in US",
        intent=jobs_live._default_intent("Fetch all PM jobs in US", "us"),
        strict=True,
        page=2,
        page_size=10,
    )
    assert result["count"] == 25
    assert result["page"] == 2
    assert result["pageSize"] == 10
    assert result["hasNextPage"] is True
    assert len(result["jobs"]) == 10
