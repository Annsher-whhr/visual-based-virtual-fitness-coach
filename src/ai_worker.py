import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional
import time

from src import ai_feedback


class AIWorker:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._latest_advice = None
        self._latest_future: Optional[Future] = None

    def submit(self, metrics: dict):
        """Submit metrics to AI in background. If a job is running, we don't queue another one."""
        with self._lock:
            if self._latest_future and not self._latest_future.done():
                return self._latest_future
            # submit a new background job
            self._latest_future = self._executor.submit(self._call_ai, metrics)
            return self._latest_future

    def _call_ai(self, metrics: dict):
        resp = ai_feedback.call_deepseek_feedback(metrics)
        advice = None
        if resp.get('ok') and isinstance(resp.get('result'), dict):
            advice = resp['result'].get('advice') or resp['result'].get('verdict')
        # store latest advice even if None to indicate completion
        with self._lock:
            self._latest_advice = advice
        return resp

    def get_latest_advice(self):
        with self._lock:
            return self._latest_advice


# singleton
_worker = AIWorker()


def submit_metrics(metrics: dict):
    return _worker.submit(metrics)


def get_latest_advice():
    return _worker.get_latest_advice()
