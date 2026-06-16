"""In-process single-flight: collapse concurrent identical requests.

Without this, N simultaneous cache misses for the same prompt all call the
provider. With it, the first request fills the cache while the rest wait on a
per-key lock, then re-check the cache instead of hitting the provider.

Locks are reference-counted and removed when idle so the map doesn't grow
unbounded across distinct keys.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class SingleFlight:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refs: dict[str, int] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, key: str):
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
                self._refs[key] = 0
            self._refs[key] += 1
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                self._refs[key] -= 1
                if self._refs[key] <= 0:
                    self._locks.pop(key, None)
                    self._refs.pop(key, None)
