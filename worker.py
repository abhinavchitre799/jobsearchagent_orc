from __future__ import annotations

import argparse

from rq import Worker

from orchestrator_queue import (
    CRITIC_QUEUE,
    DRAFTER_QUEUE,
    JOB_DISCOVERY_QUEUE,
    PLANNER_QUEUE,
    RETRIEVER_QUEUE,
    REVISER_QUEUE,
)
from queueing import get_queue, get_redis_connection

DEFAULT_QUEUES = [
    RETRIEVER_QUEUE,
    PLANNER_QUEUE,
    DRAFTER_QUEUE,
    CRITIC_QUEUE,
    REVISER_QUEUE,
    JOB_DISCOVERY_QUEUE,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ worker(s) for agent queues.")
    parser.add_argument(
        "--queues",
        default=",".join(DEFAULT_QUEUES),
        help="Comma-separated queue names. Example: retriever or planner,drafter",
    )
    parser.add_argument(
        "--redis-url",
        help="Redis URL (defaults to REDIS_URL env or redis://localhost:6379/0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_names = [name.strip() for name in args.queues.split(",") if name.strip()]
    if not queue_names:
        raise SystemExit("Provide at least one queue name via --queues.")
    connection = get_redis_connection(args.redis_url)
    queues = [get_queue(name, connection=connection) for name in queue_names]
    worker = Worker(queues=queues, connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
