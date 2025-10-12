"""
Robust debounce decorator for callbacks like GPIO interrupts.

Features:
- Leading / trailing edge control
- Optional max_wait (force-fire) to avoid indefinite deferral
- Per-key debounce (e.g., per-pin) for shared handlers
- Thread-safe (uses Lock), time.monotonic() for steady-time
- Drop-in compatible with previous debounce(wait_sec) usage
"""

from __future__ import annotations
import time
import threading
from functools import wraps
from typing import Any, Callable, Optional


def debounce(
    wait_sec: float,
    *,
    leading: bool = False,
    trailing: bool = True,
    max_wait: Optional[float] = None,
    key: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.monotonic,
):
    """
    Debounce a function so that it’s not called too frequently.

    Args:
        wait_sec: Quiet period length. Calls within this window are coalesced.
        leading:  If True, call immediately on the first call in a burst.
        trailing: If True, call after the burst “settles” for wait_sec.
        max_wait: Force a call if this many seconds have elapsed since the
                  last *invocation* (helps for steady streams).
        key:      Per-key debounce. A function that maps (*args, **kwargs) to a
                  hashable key (e.g., lambda pin, *_: pin). If None, a single
                  global bucket is used.
        clock:    Time source; default is time.monotonic().

    Notes:
        - If both leading and trailing are True:
            * first call in a burst fires immediately,
            * further calls delay a trailing fire until wait_sec after the last call,
            * but a trailing fire is skipped if there were no calls after the leading fire.
        - If only leading is True => at most one call per burst, immediate.
        - If only trailing is True => only the last call args are used.
        - If max_wait is set, a call is forced at most every max_wait seconds.
    """
    if not leading and not trailing:
        raise ValueError("At least one of leading or trailing must be True.")

    class _Bucket:
        __slots__ = (
            "lock", "timer", "last_invoke", "last_call",
            "pending", "pending_args", "pending_kwargs",
        )
        def __init__(self):
            self.lock = threading.Lock()
            self.timer: Optional[threading.Timer] = None
            self.last_invoke: Optional[float] = None  # last actual function call time
            self.last_call: Optional[float] = None    # last wrapper call time
            self.pending: bool = False
            self.pending_args = ()
            self.pending_kwargs = {}

    buckets = {}
    buckets_lock = threading.Lock()

    def _get_bucket(k: Any) -> _Bucket:
        with buckets_lock:
            b = buckets.get(k)
            if b is None:
                b = _Bucket()
                buckets[k] = b
            return b

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # derive bucket key
            k = key(*args, **kwargs) if key else None
            b = _get_bucket(k)
            now = clock()

            def _invoke_locked():
                """Call the function with the latest pending args/kwargs.
                   b.lock must be held by caller."""
                b.last_invoke = clock()
                b.pending = False
                call_args = b.pending_args
                call_kwargs = b.pending_kwargs
                # release the lock during the user call to avoid deadlocks
                b.lock.release()
                try:
                    return fn(*call_args, **call_kwargs)
                finally:
                    b.lock.acquire()

            def _schedule_trailing():
                """(Re)schedule trailing call timer from 'now'."""
                if b.timer:
                    b.timer.cancel()
                b.timer = threading.Timer(wait_sec, _trailing_fire)
                b.timer.daemon = True
                b.timer.start()

            def _trailing_fire():
                with b.lock:
                    # fire only if something is pending
                    if not b.pending:
                        return
                    # if max_wait is set and we haven't exceeded it yet AND there
                    # were no new calls, still trailing should run because timer fired
                    _invoke_locked()
                    # after trailing fire, clear timer
                    t = b.timer
                    b.timer = None
                if t:
                    try:
                        t.cancel()
                    except Exception:
                        pass

            with b.lock:
                b.last_call = now
                b.pending = True
                b.pending_args = args
                b.pending_kwargs = kwargs

                call_now = False

                # Decide leading fire
                if leading:
                    if b.last_invoke is None:
                        # first ever
                        call_now = True
                    else:
                        # if enough time has passed since last_invoke -> new burst
                        if (now - b.last_invoke) >= wait_sec and (b.timer is None):
                            call_now = True

                # Decide max_wait forced fire (never conflict with call_now)
                if (not call_now) and max_wait is not None and b.last_invoke is not None:
                    if (now - b.last_invoke) >= max_wait:
                        call_now = True

                result = None
                if call_now:
                    result = _invoke_locked()
                    # after a leading/max_wait fire, we may still want trailing if more
                    # calls keep arriving; re-arm the trailing timer if trailing is True
                    if trailing:
                        _schedule_trailing()
                else:
                    # no immediate call; manage trailing
                    if trailing:
                        _schedule_trailing()

                return result

        return wrapper
    return decorator
