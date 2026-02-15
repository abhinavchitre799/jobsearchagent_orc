from __future__ import annotations

import hashlib
import json
import os
import textwrap
import time
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx
from openai import OpenAI

DEFAULT_DISCOVERY_MODEL = "gpt-4o-mini"
DEFAULT_FETCH_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_TTL_SECONDS = 600

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}

INTENT_SYSTEM_PROMPT = """
You convert a jobs search request into structured search intent for US product roles.
Return JSON only with keys:
- role_keywords: list[string]
- location: string
- seniority: string
- must_have_terms: list[string]
- exclude_terms: list[string]
- country: string

Rules:
- Keep terms concise.
- Do not invent company names.
- If input is vague, default role_keywords to ["product manager"].
"""

NORMALIZATION_SYSTEM_PROMPT = """
You normalize raw job search provider results into strict JSON records.
Return JSON only in this shape:
{"jobs":[{"id":"","title":"","company":"","location":"","hiringManager":null,"source":"serpapi","url":"","jdText":""}]}

Rules:
- Use only evidence from provided raw payload.
- Never invent missing values.
- If URL is missing, set empty string.
- Keep jdText concise and factual from snippet/description only.
- source must be "serpapi".
"""


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def _tokenize(text: str | None) -> set[str]:
    return {t for t in _normalize(text).replace(",", " ").split(" ") if t}


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _default_intent(prompt: str, country: str) -> dict[str, Any]:
    return {
        "role_keywords": ["product manager"],
        "location": "United States" if country.lower() == "us" else country,
        "seniority": "",
        "must_have_terms": [],
        "exclude_terms": [],
        "country": country.lower(),
    }


def parse_prompt_to_intent(
    client: OpenAI,
    *,
    prompt: str,
    country: str,
    chat_model: str,
) -> dict[str, Any]:
    fallback = _default_intent(prompt, country)
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=220,
    )
    raw = (completion.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback

    role_keywords = _safe_list(parsed.get("role_keywords")) or fallback["role_keywords"]
    must_terms = _safe_list(parsed.get("must_have_terms"))
    exclude_terms = _safe_list(parsed.get("exclude_terms"))
    location = str(parsed.get("location", "")).strip() or fallback["location"]
    seniority = str(parsed.get("seniority", "")).strip()
    country_value = str(parsed.get("country", "")).strip().lower() or fallback["country"]
    return {
        "role_keywords": role_keywords,
        "location": location,
        "seniority": seniority,
        "must_have_terms": must_terms,
        "exclude_terms": exclude_terms,
        "country": country_value,
    }


def _build_serpapi_query(intent: dict[str, Any]) -> str:
    role_keywords = _safe_list(intent.get("role_keywords")) or ["product manager"]
    must_terms = _safe_list(intent.get("must_have_terms"))
    seniority = str(intent.get("seniority", "")).strip()
    location = str(intent.get("location", "")).strip() or "United States"
    base_terms = role_keywords + must_terms
    if seniority:
        base_terms.insert(0, seniority)
    query_terms = " ".join(base_terms).strip() or "product manager"
    return f"{query_terms} jobs in {location}"


def _fetch_serpapi_jobs(
    *,
    api_key: str,
    query: str,
    country: str,
    timeout_seconds: int,
    fetch_limit: int,
) -> list[dict[str, Any]]:
    params = {
        "engine": "google_jobs",
        "q": query,
        "hl": "en",
        "gl": country.lower(),
        "google_domain": "google.com",
        "api_key": api_key,
    }
    with httpx.Client(timeout=timeout_seconds) as http_client:
        response = http_client.get("https://serpapi.com/search.json", params=params)
        response.raise_for_status()
    payload = response.json()
    jobs = payload.get("jobs_results", [])
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs[: max(1, min(fetch_limit, 50))] if isinstance(job, dict)]


def _extract_url(raw_job: dict[str, Any]) -> str:
    for key in ("job_google_link", "share_link", "link"):
        val = str(raw_job.get(key, "")).strip()
        if val:
            return val
    related_links = raw_job.get("related_links")
    if isinstance(related_links, list):
        for item in related_links:
            if not isinstance(item, dict):
                continue
            val = str(item.get("link", "")).strip()
            if val:
                return val
    apply_options = raw_job.get("apply_options")
    if isinstance(apply_options, list):
        for item in apply_options:
            if not isinstance(item, dict):
                continue
            val = str(item.get("link", "")).strip()
            if val:
                return val
    return ""


def _fallback_normalize(raw_jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_jobs:
        title = str(raw.get("title", "")).strip()
        company = str(raw.get("company_name", "") or raw.get("company", "")).strip()
        location = str(raw.get("location", "")).strip() or "US"
        url = _extract_url(raw)
        description = str(raw.get("description", "") or raw.get("snippet", "")).strip()
        job_id = str(raw.get("job_id", "")).strip() or ""
        normalized.append(
            {
                "id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "hiringManager": None,
                "source": "serpapi",
                "url": url,
                "jdText": description,
            }
        )
    return normalized


def normalize_jobs_with_llm(
    client: OpenAI,
    *,
    raw_jobs: Sequence[dict[str, Any]],
    intent: dict[str, Any],
    chat_model: str,
) -> list[dict[str, Any]]:
    compact_jobs = []
    for raw in raw_jobs:
        compact_jobs.append(
            {
                "job_id": raw.get("job_id"),
                "title": raw.get("title"),
                "company_name": raw.get("company_name"),
                "location": raw.get("location"),
                "description": raw.get("description"),
                "snippet": raw.get("snippet"),
                "related_links": raw.get("related_links"),
                "apply_options": raw.get("apply_options"),
                "job_google_link": raw.get("job_google_link"),
                "share_link": raw.get("share_link"),
            }
        )
    prompt = textwrap.dedent(
        f"""
        Intent:
        {json.dumps(intent, indent=2)}

        Raw provider jobs:
        {json.dumps(compact_jobs, indent=2)}
        """
    ).strip()
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": NORMALIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2500,
    )
    raw = (completion.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_normalize(raw_jobs)
    if not isinstance(parsed, dict):
        return _fallback_normalize(raw_jobs)
    jobs = parsed.get("jobs")
    if not isinstance(jobs, list):
        return _fallback_normalize(raw_jobs)
    cleaned: list[dict[str, Any]] = []
    for item in jobs:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "id": str(item.get("id", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "company": str(item.get("company", "")).strip(),
                "location": str(item.get("location", "")).strip(),
                "hiringManager": str(item.get("hiringManager", "")).strip() or None,
                "source": "serpapi",
                "url": str(item.get("url", "")).strip(),
                "jdText": str(item.get("jdText", "")).strip(),
            }
        )
    return cleaned if cleaned else _fallback_normalize(raw_jobs)


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _stable_id(candidate: str) -> str:
    return hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:16]


def _rank_score(job: dict[str, Any], intent: dict[str, Any], prompt: str) -> int:
    title_tokens = _tokenize(str(job.get("title", "")))
    body_tokens = _tokenize(
        " ".join(
            [
                str(job.get("title", "")),
                str(job.get("company", "")),
                str(job.get("location", "")),
                str(job.get("jdText", "")),
            ]
        )
    )
    role_tokens = set()
    for term in _safe_list(intent.get("role_keywords")):
        role_tokens |= _tokenize(term)
    must_tokens = set()
    for term in _safe_list(intent.get("must_have_terms")):
        must_tokens |= _tokenize(term)
    location_tokens = _tokenize(str(intent.get("location", "")))
    prompt_tokens = _tokenize(prompt)

    score = 0
    for token in role_tokens:
        if token in title_tokens:
            score += 4
        elif token in body_tokens:
            score += 2
    for token in must_tokens:
        if token in body_tokens:
            score += 2
    for token in prompt_tokens:
        if token in title_tokens:
            score += 2
    if location_tokens and any(token in body_tokens for token in location_tokens):
        score += 3
    return score


def strict_verify_rank_paginate(
    *,
    jobs: Sequence[dict[str, Any]],
    prompt: str,
    intent: dict[str, Any],
    strict: bool,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    exclude_tokens = set()
    for term in _safe_list(intent.get("exclude_terms")):
        exclude_tokens |= _tokenize(term)
    for raw_job in jobs:
        title = str(raw_job.get("title", "")).strip()
        company = str(raw_job.get("company", "")).strip()
        location = str(raw_job.get("location", "")).strip() or "US"
        url = str(raw_job.get("url", "")).strip()
        jd_text = str(raw_job.get("jdText", "")).strip()
        full_tokens = _tokenize(" ".join([title, company, location, jd_text]))
        if exclude_tokens and any(tok in full_tokens for tok in exclude_tokens):
            continue
        if strict:
            if not title or not company or not url or not jd_text:
                continue
            if not _is_valid_url(url):
                continue

        candidate_id = str(raw_job.get("id", "")).strip()
        stable_source = candidate_id or url or f"{company}-{title}-{location}"
        job_id = _stable_id(stable_source)
        if url and url in seen_urls:
            continue
        if job_id in seen_ids:
            continue
        seen_ids.add(job_id)
        if url:
            seen_urls.add(url)
        deduped.append(
            {
                "id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "hiringManager": raw_job.get("hiringManager"),
                "source": "serpapi",
                "url": url,
                "jdText": jd_text,
            }
        )

    ranked = sorted(
        deduped,
        key=lambda job: (
            _rank_score(job, intent, prompt),
            _normalize(str(job.get("title", ""))),
            _normalize(str(job.get("company", ""))),
        ),
        reverse=True,
    )
    page_value = max(1, page)
    size_value = max(1, min(page_size, 25))
    start = (page_value - 1) * size_value
    end = start + size_value
    jobs_slice = ranked[start:end]
    return {
        "count": len(ranked),
        "page": page_value,
        "pageSize": size_value,
        "hasNextPage": end < len(ranked),
        "jobs": jobs_slice,
        "source": "serpapi",
        "_ranked_all": ranked,
    }


def _cache_key(prompt: str, country: str, strict: bool) -> str:
    return f"{_normalize(prompt)}::{country.lower()}::{int(strict)}"


def discover_jobs_live(
    client: OpenAI,
    *,
    prompt: str,
    country: str,
    page: int,
    page_size: int,
    strict: bool,
    serpapi_api_key: str,
    chat_model: str = DEFAULT_DISCOVERY_MODEL,
) -> dict[str, Any]:
    fetch_limit = int(os.getenv("JOB_DISCOVERY_FETCH_LIMIT", str(DEFAULT_FETCH_LIMIT)))
    timeout_seconds = int(os.getenv("JOB_DISCOVERY_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    cache_ttl = int(os.getenv("JOB_DISCOVERY_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS)))

    cache_key = _cache_key(prompt, country, strict)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and cached[0] > now:
        ranked_jobs = cached[1]
        start = (max(1, page) - 1) * max(1, min(page_size, 25))
        end = start + max(1, min(page_size, 25))
        return {
            "count": len(ranked_jobs),
            "page": max(1, page),
            "pageSize": max(1, min(page_size, 25)),
            "hasNextPage": end < len(ranked_jobs),
            "jobs": ranked_jobs[start:end],
            "source": "serpapi",
        }

    intent = parse_prompt_to_intent(
        client,
        prompt=prompt,
        country=country,
        chat_model=chat_model,
    )
    query = _build_serpapi_query(intent)
    raw_jobs = _fetch_serpapi_jobs(
        api_key=serpapi_api_key,
        query=query,
        country=country,
        timeout_seconds=timeout_seconds,
        fetch_limit=fetch_limit,
    )
    normalized_jobs = normalize_jobs_with_llm(
        client,
        raw_jobs=raw_jobs,
        intent=intent,
        chat_model=chat_model,
    )
    result = strict_verify_rank_paginate(
        jobs=normalized_jobs,
        prompt=prompt,
        intent=intent,
        strict=strict,
        page=page,
        page_size=page_size,
    )
    ranked_all = result.pop("_ranked_all")
    _CACHE[cache_key] = (now + max(1, cache_ttl), ranked_all)
    return result
