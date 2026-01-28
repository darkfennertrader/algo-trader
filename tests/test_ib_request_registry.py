import threading
import time

from algo_trader.providers.historical.ib import IbRequestRegistry


def test_wait_all_returns_true_when_complete() -> None:
    registry = IbRequestRegistry(inflight_requests=threading.Semaphore(1))
    registry.register(1, "AUD")
    registry.inflight_requests.acquire()
    registry.mark_done(1)
    assert registry.wait_all(check_interval=0.01)


def test_wait_all_aborts_when_event_set() -> None:
    registry = IbRequestRegistry(inflight_requests=threading.Semaphore(1))
    registry.register(1, "AUD")
    abort_event = threading.Event()

    def _trigger_abort() -> None:
        time.sleep(0.05)
        abort_event.set()

    thread = threading.Thread(target=_trigger_abort, daemon=True)
    thread.start()
    assert not registry.wait_all(abort_event=abort_event, check_interval=0.01)
    thread.join(timeout=1.0)
