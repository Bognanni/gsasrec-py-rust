"""
HTTP transport for the engine — backed by aiohttp.

Each route's request_builder uses this in `make_client(config, num_clients)`:

    from contextlib import asynccontextmanager
    from http_transport import make_http_client

    @asynccontextmanager
    async def make_client(config, num_clients):
        c = config['client']
        async with make_http_client(
            url=get_endpoint_url(config),
            get_headers_fn=lambda: get_headers(config),
            num_clients=num_clients,
            connect_timeout=c.get('connect_timeout', 3),
            read_timeout=c.get('read_timeout', 60),
        ) as invoke:
            yield invoke
"""

import time
from contextlib import asynccontextmanager

import aiohttp


@asynccontextmanager
async def make_http_client(url: str, get_headers_fn, num_clients: int,
                           connect_timeout: float = 3,
                           read_timeout: float = 60):
    """
    Open one aiohttp session with a connection pool of `num_clients` and yield
    a single `invoke(payload) -> latency_ms` coroutine.
    """
    connector = aiohttp.TCPConnector(limit=num_clients)
    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=connect_timeout,
        sock_read=read_timeout,
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def invoke(payload: dict) -> float:
            t0 = time.time()
            async with session.post(url, json=payload, headers=get_headers_fn()) as resp:
                body = await resp.read()
                if resp.status != 200:
                    snippet = body[:500].decode('utf-8', errors='replace')
                    raise RuntimeError(f'HTTP {resp.status}: {snippet}')
            return (time.time() - t0) * 1000.0

        yield invoke