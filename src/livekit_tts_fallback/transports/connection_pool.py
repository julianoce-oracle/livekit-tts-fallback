from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from ..errors import ConfigurationError


class ReusableConnection(Protocol):
    @property
    def is_open(self) -> bool: ...

    async def close(self) -> None: ...


ConnectionT = TypeVar("ConnectionT", bound=ReusableConnection)


@dataclass(frozen=True, slots=True)
class ConnectionPoolConfig:
    min_size: int = 0
    max_size: int = 3
    acquire_timeout_s: float = 3.0
    session_ttl_s: float = 540.0
    max_uses: int | None = None

    def __post_init__(self) -> None:
        if self.min_size < 0:
            raise ConfigurationError("pool min_size cannot be negative")
        if self.max_size < 1:
            raise ConfigurationError("pool max_size must be at least one")
        if self.min_size > self.max_size:
            raise ConfigurationError("pool min_size cannot exceed max_size")
        if self.acquire_timeout_s <= 0:
            raise ConfigurationError("pool acquire_timeout_s must be positive")
        if self.session_ttl_s <= 0:
            raise ConfigurationError("pool session_ttl_s must be positive")
        if self.max_uses is not None and self.max_uses < 1:
            raise ConfigurationError("pool max_uses must be positive when set")


@dataclass(slots=True)
class _PoolEntry(Generic[ConnectionT]):
    entry_id: str
    connection: ConnectionT
    created_at: float
    uses: int = 0
    leased: bool = False


@dataclass(frozen=True, slots=True)
class ConnectionLease(Generic[ConnectionT]):
    entry_id: str
    connection: ConnectionT
    reused: bool


class AsyncConnectionPool(Generic[ConnectionT]):
    """Small event-loop-local pool for providers with reusable sessions."""

    def __init__(
        self,
        factory: Callable[[], Awaitable[ConnectionT]],
        *,
        config: ConnectionPoolConfig | None = None,
    ) -> None:
        self._factory = factory
        self.config = config or ConnectionPoolConfig()
        self._condition = asyncio.Condition()
        self._idle: deque[_PoolEntry[ConnectionT]] = deque()
        self._entries: dict[str, _PoolEntry[ConnectionT]] = {}
        self._reserved = 0
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._entries) + self._reserved

    @property
    def idle_count(self) -> int:
        return len(self._idle)

    async def start(self) -> None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("connection pool is closed")
            missing = max(0, self.config.min_size - self.size)
            self._reserved += missing

        if missing == 0:
            return

        results = await asyncio.gather(
            *(self._factory() for _ in range(missing)),
            return_exceptions=True,
        )
        first_error: BaseException | None = None
        async with self._condition:
            self._reserved -= missing
            for result in results:
                if isinstance(result, BaseException):
                    first_error = first_error or result
                    continue
                entry = self._new_entry(result)
                self._entries[entry.entry_id] = entry
                self._idle.append(entry)
            self._condition.notify_all()

        if first_error is not None:
            raise first_error

    async def acquire(self, *, timeout_s: float | None = None) -> ConnectionLease[ConnectionT]:
        timeout_s = timeout_s or self.config.acquire_timeout_s
        deadline = asyncio.get_running_loop().time() + timeout_s

        while True:
            stale: list[_PoolEntry[ConnectionT]] = []
            create_new = False
            async with self._condition:
                if self._closed:
                    raise RuntimeError("connection pool is closed")

                while self._idle:
                    entry = self._idle.popleft()
                    if self._is_reusable(entry):
                        entry.leased = True
                        entry.uses += 1
                        return ConnectionLease(
                            entry_id=entry.entry_id,
                            connection=entry.connection,
                            reused=entry.uses > 1,
                        )
                    self._entries.pop(entry.entry_id, None)
                    stale.append(entry)

                if self.size < self.config.max_size:
                    self._reserved += 1
                    create_new = True
                else:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for a reusable connection")
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except TimeoutError as exc:
                        raise TimeoutError("timed out waiting for a reusable connection") from exc

            await self._close_entries(stale)
            if not create_new:
                continue

            try:
                connection = await self._factory()
            except BaseException:
                async with self._condition:
                    self._reserved -= 1
                    self._condition.notify_all()
                raise

            async with self._condition:
                self._reserved -= 1
                if self._closed:
                    close_after_unlock = True
                else:
                    entry = self._new_entry(connection)
                    entry.leased = True
                    entry.uses = 1
                    self._entries[entry.entry_id] = entry
                    self._condition.notify_all()
                    return ConnectionLease(
                        entry_id=entry.entry_id,
                        connection=entry.connection,
                        reused=False,
                    )

            if close_after_unlock:
                await connection.close()
                raise RuntimeError("connection pool is closed")

    async def release(self, lease: ConnectionLease[ConnectionT], *, healthy: bool) -> None:
        close_entry: _PoolEntry[ConnectionT] | None = None
        async with self._condition:
            entry = self._entries.get(lease.entry_id)
            if entry is None:
                return

            entry.leased = False
            if not self._closed and healthy and self._is_reusable(entry):
                self._idle.append(entry)
            else:
                self._entries.pop(entry.entry_id, None)
                close_entry = entry
            self._condition.notify_all()

        if close_entry is not None:
            await self._close_entries([close_entry])

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
            self._idle.clear()
            self._condition.notify_all()
        await self._close_entries(entries)

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "size": self.size,
            "idle": self.idle_count,
            "leased": sum(1 for entry in self._entries.values() if entry.leased),
            "closed": self._closed,
        }

    def _new_entry(self, connection: ConnectionT) -> _PoolEntry[ConnectionT]:
        return _PoolEntry(
            entry_id=uuid.uuid4().hex,
            connection=connection,
            created_at=time.monotonic(),
        )

    def _is_reusable(self, entry: _PoolEntry[ConnectionT]) -> bool:
        if not entry.connection.is_open:
            return False
        if time.monotonic() - entry.created_at >= self.config.session_ttl_s:
            return False
        return self.config.max_uses is None or entry.uses < self.config.max_uses

    @staticmethod
    async def _close_entries(entries: list[_PoolEntry[ConnectionT]]) -> None:
        for entry in entries:
            with contextlib.suppress(Exception):
                await entry.connection.close()
