from __future__ import annotations

from typing import Any


_JOB_SEED: list[dict[str, Any]] = [
    {
        "id": "pm-001",
        "title": "Senior Product Manager, Growth",
        "company": "Northwind Labs",
        "location": "Remote (US)",
        "hiringManager": "Jordan Lee",
        "source": "seed",
        "url": "https://example.com/jobs/pm-001",
        "jdText": (
            "Northwind Labs is hiring a Senior Product Manager to lead growth experimentation, "
            "pricing tests, and lifecycle onboarding. You will partner with design, analytics, "
            "and engineering to improve activation and retention across web and mobile."
        ),
    },
    {
        "id": "pm-002",
        "title": "Product Manager, Platform",
        "company": "Contoso Cloud",
        "location": "San Francisco, CA",
        "hiringManager": "Priya Raman",
        "source": "seed",
        "url": "https://example.com/jobs/pm-002",
        "jdText": (
            "Own API platform roadmap and developer experience for Contoso Cloud. Drive product "
            "strategy for identity, API governance, and platform reliability in partnership with "
            "engineering and enterprise customers."
        ),
    },
    {
        "id": "pm-003",
        "title": "Principal Product Manager, AI Assistants",
        "company": "Fabrikam AI",
        "location": "New York, NY",
        "hiringManager": "Evan Brooks",
        "source": "seed",
        "url": "https://example.com/jobs/pm-003",
        "jdText": (
            "Lead roadmap for AI assistant workflows including prompt quality, evaluation metrics, "
            "and enterprise controls. Collaborate with data science and design teams to ship "
            "high-usage assistant capabilities."
        ),
    },
    {
        "id": "pm-004",
        "title": "Product Manager, Data & Analytics",
        "company": "Adventure Works",
        "location": "Austin, TX",
        "hiringManager": "",
        "source": "seed",
        "url": "https://example.com/jobs/pm-004",
        "jdText": (
            "Define analytics product requirements and self-serve reporting experiences. Partner "
            "with BI engineers to improve data model quality and executive dashboards."
        ),
    },
    {
        "id": "pm-005",
        "title": "Group Product Manager, B2B SaaS",
        "company": "Litware",
        "location": "Remote (US)",
        "hiringManager": "Taylor Kim",
        "source": "seed",
        "url": "https://example.com/jobs/pm-005",
        "jdText": (
            "Own product strategy for B2B collaboration suite. Manage a team of PMs, align GTM "
            "with sales enablement, and improve enterprise adoption with measurable outcomes."
        ),
    },
    {
        "id": "pm-006",
        "title": "Associate Product Manager",
        "company": "Woodgrove Bank",
        "location": "Chicago, IL",
        "hiringManager": "Hiring Team",
        "source": "seed",
        "url": "https://example.com/jobs/pm-006",
        "jdText": (
            "Support roadmap execution for digital banking features. Work with compliance, data, "
            "and engineering teams to deliver secure customer-facing capabilities."
        ),
    },
]


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def _tokenize(text: str | None) -> set[str]:
    return {tok for tok in _normalize(text).replace(",", " ").split(" ") if tok}


def _score_job(job: dict[str, Any], query_tokens: set[str]) -> int:
    haystack = _normalize(
        " ".join(
            [
                str(job.get("title", "")),
                str(job.get("company", "")),
                str(job.get("location", "")),
                str(job.get("jdText", "")),
            ]
        )
    )
    title_tokens = _tokenize(str(job.get("title", "")))
    score = 0
    if not query_tokens:
        return 1
    for token in query_tokens:
        if token in title_tokens:
            score += 3
        elif token in haystack:
            score += 1
    return score


def discover_jobs(
    *,
    query: str | None,
    location: str | None = None,
    company: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize(query or "product manager")
    location_tokens = _tokenize(location)
    company_tokens = _tokenize(company)

    scored: list[tuple[int, dict[str, Any]]] = []
    for job in _JOB_SEED:
        location_text = _normalize(str(job.get("location", "")))
        company_text = _normalize(str(job.get("company", "")))
        if location_tokens and not all(tok in location_text for tok in location_tokens):
            continue
        if company_tokens and not all(tok in company_text for tok in company_tokens):
            continue
        score = _score_job(job, query_tokens)
        if score <= 0:
            continue
        scored.append((score, job))

    scored.sort(key=lambda item: item[0], reverse=True)
    trimmed = scored[: max(1, min(limit, 50))]
    return [job for _, job in trimmed]


def discover_jobs_seed(
    *,
    prompt: str,
    page: int,
    page_size: int,
    country: str,
    strict: bool,
) -> dict[str, Any]:
    _ = country, strict
    ranked = discover_jobs(query=prompt or "Product Manager", limit=50)
    page_value = max(1, page)
    size_value = max(1, min(page_size, 25))
    start = (page_value - 1) * size_value
    end = start + size_value
    page_jobs = ranked[start:end]
    return {
        "count": len(ranked),
        "page": page_value,
        "pageSize": size_value,
        "hasNextPage": end < len(ranked),
        "jobs": page_jobs,
        "source": "seed",
    }
