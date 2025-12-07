from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from rag import (
    build_llm_message,
    retrieve_chunks_with_embeddings,
    split_into_chunks,
    truncate_text,
    DEFAULT_MAX_CHARS,
)

DEFAULT_API_KEY = "REPLACE_WITH_OPENAI_KEY"


def load_text(path: Path) -> str:
    """Load text from a file, normalizing newlines."""
    raw = path.read_text(encoding="utf-8")
    return "\n".join(line.rstrip() for line in raw.splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LinkedIn outreach note using OpenAI embeddings + chat (RAG)."
    )
    parser.add_argument("--resume", required=True, type=Path, help="Path to your resume text file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-text", help="Inline JD or LinkedIn message text.")
    source.add_argument("--input-file", type=Path, help="File containing the JD or LinkedIn message.")
    parser.add_argument("--hm-note", help="Optional note from hiring manager to blend with the JD.")
    parser.add_argument("--name", required=True, help="Your name to sign the message.")
    parser.add_argument("--role", help="Role you're targeting, if known.")
    parser.add_argument("--company", help="Company name, if known.")
    parser.add_argument("--hiring-manager", help="Hiring manager or contact name, optional.")
    parser.add_argument(
        "--output-type",
        choices=["message", "cover-letter"],
        default="message",
        help="Generate a LinkedIn message or a cover letter (default: message).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many resume snippets to ground the message with (default: 3).",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model to use.",
    )
    parser.add_argument(
        "--chat-model",
        default="gpt-4o-mini",
        help="OpenAI chat model to draft the message.",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (defaults to env OPENAI_API_KEY).",
    )
    return parser.parse_args()


def read_query_text(args: argparse.Namespace) -> str:
    parts = []
    if args.input_text:
        parts.append(args.input_text.strip())
    if args.input_file:
        parts.append(load_text(args.input_file).strip())
    if args.hm_note:
        parts.append(args.hm_note.strip())
    if not parts:
        raise ValueError("No input text provided.")
    combined = "\n\n".join(parts)
    return truncate_text(combined)


def token_estimate(*texts: str) -> int:
    """Rough token estimate using 4 chars per token heuristic."""
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // 4)


def main() -> None:
    args = parse_args()
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    if not api_key or api_key == "REPLACE_WITH_OPENAI_KEY":
        raise SystemExit("Set OPENAI_API_KEY env, pass --openai-api-key, or set DEFAULT_API_KEY in agent.py.")

    client = OpenAI(api_key=api_key)
    resume_text = truncate_text(load_text(args.resume), DEFAULT_MAX_CHARS)
    resume_chunks = split_into_chunks(resume_text)
    if not resume_chunks or not resume_chunks[0].strip():
        raise SystemExit("Resume appears empty after parsing.")

    query_text = truncate_text(read_query_text(args), DEFAULT_MAX_CHARS // 2)
    approx_tokens = token_estimate(resume_text, query_text)
    print(f"[info] Estimated prompt tokens: ~{approx_tokens}", flush=True)
    role_value = args.role or "Product Manager"
    retrieved = retrieve_chunks_with_embeddings(
        client,
        resume_chunks,
        query_text,
        embedding_model=args.embedding_model,
        top_k=max(1, args.top_k),
    )
    message = build_llm_message(
        client,
        candidate=args.name,
        role=role_value,
        company=args.company,
        hiring_manager=args.hiring_manager,
        query=query_text,
        retrieved_chunks=retrieved,
        chat_model=args.chat_model,
        output_type=args.output_type,
    )
    print("\n" + message)


if __name__ == "__main__":
    main()
