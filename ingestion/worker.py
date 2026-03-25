
from __future__ import annotations
import asyncio
import logging

logger = logging.getLogger(__name__)


class LocalQueueWorker:
    """
    Local dev replacement for the Azure Service Bus worker.
    Uses asyncio.Queue — same concurrency model, zero Azure dependency.

    Usage in main.py:
        worker = LocalQueueWorker(pipeline_fn=_make_pipeline_fn(app))
        asyncio.create_task(worker.run())
        # To enqueue a message:
        await app.state.queue.put(msg_dict)
    """

    def __init__(self, pipeline_fn, max_concurrent: int = 5):
        self.queue: asyncio.Queue = asyncio.Queue()
        self._pipeline = pipeline_fn
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("LocalQueueWorker started.")
        while self._running:
            try:
                # Wait for a message, timeout every 1s to check _running
                msg = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            asyncio.create_task(self._handle(msg))

    async def _handle(self, msg: dict) -> None:
        async with self._semaphore:
            try:
                logger.info(
                    "Processing: applicant=%s job=%s",
                    msg.get("applicant_id"), msg.get("job_id"),
                )
                await self._pipeline(msg)
                self.queue.task_done()
                logger.info("Completed: %s", msg.get("applicant_id"))
            except Exception as exc:
                logger.exception("Pipeline error: %s", exc)
                self.queue.task_done()

    async def stop(self) -> None:
        self._running = False