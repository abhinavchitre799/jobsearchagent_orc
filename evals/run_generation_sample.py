from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


@dataclass
class CheckResult:
    ok: bool
    message: str


def _word_count(text: str) -> int:
    return len([t for t in re.split(r"\s+", text.strip()) if t])


def _count_paragraphs(text: str) -> int:
    # Paragraphs separated by blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paras)


def _contains_case_insensitive(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def evaluate_output(case: Dict[str, Any], output: str) -> Tuple[bool, List[CheckResult]]:
    expect = case.get("expect") or {}
    results: List[CheckResult] = []

    wc = _word_count(output)
    max_words = int(expect.get("maxWords", 999999))
    results.append(
        CheckResult(ok=wc <= max_words, message=f"word_count {wc} <= {max_words}")
    )

    must_include = list(expect.get("mustInclude") or [])
    for token in must_include:
        ok = _contains_case_insensitive(output, token)
        results.append(CheckResult(ok=ok, message=f"must_include {token!r}"))

    must_not = list(expect.get("mustNotInclude") or [])
    for token in must_not:
        ok = not _contains_case_insensitive(output, token)
        results.append(CheckResult(ok=ok, message=f"must_not_include {token!r}"))

    min_markers = int(expect.get("minFromResumeMarkers", 0))
    markers = len(re.findall(r"\(from resume\)", output, flags=re.I))
    results.append(
        CheckResult(ok=markers >= min_markers, message=f"(from resume) markers {markers} >= {min_markers}")
    )

    output_type = case.get("outputType", "message")
    hiring_manager = (case.get("hiringManager") or "").strip()
    company = (case.get("company") or "").strip()
    candidate = (case.get("name") or "").strip()

    if output_type == "message":
        # Basic greeting check.
        ok = bool(re.search(r"^\s*(hi|hello|dear)\b", output, flags=re.I))
        results.append(CheckResult(ok=ok, message="starts_with_greeting"))
        if hiring_manager:
            # Expect the hiring manager name to appear near the top.
            top = " ".join(output.split()[:40])
            ok = _contains_case_insensitive(top, hiring_manager)
            results.append(CheckResult(ok=ok, message="mentions_hiring_manager_near_top"))
        if company:
            ok = _contains_case_insensitive(output, company)
            results.append(CheckResult(ok=ok, message="mentions_company"))

    if output_type == "cover-letter":
        ok = _contains_case_insensitive(output, "sincerely")
        results.append(CheckResult(ok=ok, message="contains_sincerely"))
        if candidate:
            ok = _contains_case_insensitive(output, candidate)
            results.append(CheckResult(ok=ok, message="contains_candidate_name"))
        if company:
            ok = _contains_case_insensitive(output, company)
            results.append(CheckResult(ok=ok, message="mentions_company"))
        min_paras = int(expect.get("minParagraphs", 0))
        paras = _count_paragraphs(output)
        results.append(CheckResult(ok=paras >= min_paras, message=f"paragraphs {paras} >= {min_paras}"))

    ok_all = all(r.ok for r in results)
    return ok_all, results


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Sample evals for /generate (message + cover letter).")
    ap.add_argument("--api-base", default="http://localhost:8000", help="API base URL")
    ap.add_argument(
        "--cases",
        default=str(Path("evals/cases/generation_sample.jsonl")),
        help="Path to JSONL cases file",
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Cases file not found: {cases_path}", file=sys.stderr)
        return 2

    cases: List[Dict[str, Any]] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))

    client = httpx.Client(timeout=args.timeout)
    any_failed = False

    for case in cases:
        case_id = case.get("id", "<missing-id>")
        payload = {k: v for k, v in case.items() if k not in {"id", "expect"}}

        try:
            resp = client.post(f"{args.api_base}/generate", json=payload)
        except Exception as exc:
            any_failed = True
            print(f"[FAIL] {case_id}: request_error {exc}")
            continue

        if resp.status_code != 200:
            any_failed = True
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text[:200]
            print(f"[FAIL] {case_id}: http_{resp.status_code} {detail}")
            continue

        data = resp.json()
        output = str(data.get("message", "")).strip()
        ok, checks = evaluate_output(case, output)
        if ok:
            print(f"[PASS] {case_id} (words={_word_count(output)})")
        else:
            any_failed = True
            print(f"[FAIL] {case_id} (words={_word_count(output)})")
            for cr in checks:
                if not cr.ok:
                    print(f"  - {cr.message}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

