from __future__ import annotations

import pytest

from livekit_tts_fallback.transports import AsyncConnectionPool, ConnectionPoolConfig


class FakeConnection:
    def __init__(self) -> None:
        self.open = True
        self.close_calls = 0

    @property
    def is_open(self) -> bool:
        return self.open

    async def close(self) -> None:
        self.open = False
        self.close_calls += 1


@pytest.mark.asyncio
async def test_reuses_a_healthy_connection() -> None:
    created: list[FakeConnection] = []

    async def factory() -> FakeConnection:
        connection = FakeConnection()
        created.append(connection)
        return connection

    pool = AsyncConnectionPool(
        factory,
        config=ConnectionPoolConfig(min_size=1, max_size=2),
    )
    await pool.start()

    first = await pool.acquire()
    assert first.reused is False
    await pool.release(first, healthy=True)

    second = await pool.acquire()
    assert second.reused is True
    assert second.connection is first.connection
    await pool.release(second, healthy=True)
    assert len(created) == 1

    await pool.aclose()
    assert created[0].close_calls == 1


@pytest.mark.asyncio
async def test_discards_an_unhealthy_connection() -> None:
    created: list[FakeConnection] = []

    async def factory() -> FakeConnection:
        connection = FakeConnection()
        created.append(connection)
        return connection

    pool = AsyncConnectionPool(factory, config=ConnectionPoolConfig(max_size=1))
    first = await pool.acquire()
    await pool.release(first, healthy=False)
    second = await pool.acquire()

    assert second.connection is not first.connection
    assert created[0].close_calls == 1
    await pool.release(second, healthy=True)
    await pool.aclose()


@pytest.mark.asyncio
async def test_rotates_connection_after_max_uses() -> None:
    async def factory() -> FakeConnection:
        return FakeConnection()

    pool = AsyncConnectionPool(
        factory,
        config=ConnectionPoolConfig(max_size=1, max_uses=1),
    )
    first = await pool.acquire()
    await pool.release(first, healthy=True)
    second = await pool.acquire()

    assert second.connection is not first.connection
    assert first.connection.close_calls == 1
    await pool.release(second, healthy=True)
    await pool.aclose()
