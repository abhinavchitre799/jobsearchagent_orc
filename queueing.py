from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis import Redis
    from rq import Queue
    from rq.job import Job

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_JOB_TIMEOUT_SECONDS = 180
DEFAULT_RESULT_TTL_SECONDS = 600
DEFAULT_WAIT_TIMEOUT_SECONDS = 180
DEFAULT_POLL_SECONDS = 0.2


class QueueTaskError(RuntimeError):
    """Raised when a queued task fails or returns an invalid state."""


def get_redis_connection(redis_url: str | None = None) -> Redis:
    try:
        from redis import Redis
    except ModuleNotFoundError as exc:
        raise RuntimeError("Queue runtime requires 'redis'. Install dependencies from requirements.txt.") from exc
    return Redis.from_url(redis_url or os.getenv("REDIS_URL", DEFAULT_REDIS_URL))


def get_queue(
    queue_name: str,
    *,
    connection: Redis,
    default_timeout: int | None = None,
) -> Queue:
    try:
        from rq import Queue
    except ModuleNotFoundError as exc:
        raise RuntimeError("Queue runtime requires 'rq'. Install dependencies from requirements.txt.") from exc
    timeout_value = default_timeout or int(
        os.getenv("AGENT_JOB_TIMEOUT_SECONDS", str(DEFAULT_JOB_TIMEOUT_SECONDS))
    )
    return Queue(name=queue_name, connection=connection, default_timeout=timeout_value)


def enqueue_task(
    queue: Queue,
    task_path: str,
    *,
    kwargs: dict[str, Any],
    job_timeout: int | None = None,
    result_ttl: int | None = None,
    description: str | None = None,
) -> Job:
    timeout_value = job_timeout or int(
        os.getenv("AGENT_JOB_TIMEOUT_SECONDS", str(DEFAULT_JOB_TIMEOUT_SECONDS))
    )
    result_ttl_value = result_ttl or int(
        os.getenv("AGENT_RESULT_TTL_SECONDS", str(DEFAULT_RESULT_TTL_SECONDS))
    )
    return queue.enqueue(
        task_path,
        kwargs=kwargs,
        job_timeout=timeout_value,
        result_ttl=result_ttl_value,
        description=description,
    )


def wait_for_job(
    *,
    job: Job,
    timeout_seconds: int | None = None,
    poll_seconds: float | None = None,
) -> Any:
    wait_timeout = timeout_seconds or int(
        os.getenv("AGENT_WAIT_TIMEOUT_SECONDS", str(DEFAULT_WAIT_TIMEOUT_SECONDS))
    )
    poll = poll_seconds or float(os.getenv("AGENT_POLL_SECONDS", str(DEFAULT_POLL_SECONDS)))
    deadline = time.monotonic() + wait_timeout
    while True:
        status = job.get_status(refresh=True)
        if status == "finished":
            return job.result
        if status == "failed":
            details = job.exc_info or "No worker traceback available."
            raise QueueTaskError(f"Job {job.id} failed: {details}")
        if status in {"stopped", "canceled"}:
            raise QueueTaskError(f"Job {job.id} ended with status '{status}'.")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for job {job.id} after {wait_timeout}s.")
        time.sleep(poll)
