"""RequestCache — per-request in-process snapshot cache for GraphQL resolvers."""

import asyncio
import gc

import pytest
from unifi_api.graphql.context import RequestCache


@pytest.mark.asyncio
async def test_request_cache_dedupes_concurrent_fetches() -> None:
    """Two concurrent get_or_fetch calls with the same key produce one fetch."""
    cache = RequestCache()
    fetch_count = 0

    async def _slow_fetch():
        nonlocal fetch_count
        fetch_count += 1
        await asyncio.sleep(0.01)
        return [{"id": 1}, {"id": 2}]

    a, b = await asyncio.gather(
        cache.get_or_fetch("key1", _slow_fetch),
        cache.get_or_fetch("key1", _slow_fetch),
    )
    assert a == b == [{"id": 1}, {"id": 2}]
    assert fetch_count == 1


@pytest.mark.asyncio
async def test_request_cache_isolates_distinct_keys() -> None:
    cache = RequestCache()

    async def _fetch_a():
        return "a"

    async def _fetch_b():
        return "b"

    assert await cache.get_or_fetch("k1", _fetch_a) == "a"
    assert await cache.get_or_fetch("k2", _fetch_b) == "b"


@pytest.mark.asyncio
async def test_request_cache_replays_stored_value_on_repeat() -> None:
    cache = RequestCache()
    calls = 0

    async def _fetch():
        nonlocal calls
        calls += 1
        return calls

    assert await cache.get_or_fetch("k", _fetch) == 1
    assert await cache.get_or_fetch("k", _fetch) == 1  # cached, not refetched
    assert calls == 1


@pytest.mark.asyncio
async def test_request_cache_consumes_unobserved_failed_future() -> None:
    """A sole caller still receives the error without leaking an asyncio warning."""
    cache = RequestCache()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    async def _failing_fetch() -> None:
        raise RuntimeError("expected failure")

    try:
        with pytest.raises(RuntimeError, match="expected failure"):
            await cache.get_or_fetch("failing-key", _failing_fetch)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert loop_errors == []
