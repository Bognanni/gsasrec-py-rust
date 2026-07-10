"""
Transport-agnostic load test engine.

The engine runs a two-phase Poisson load test and never touches a transport
library directly. The transport is injected via `make_client_fn`, which is
an async context manager that yields a single coroutine:

    invoke(payload: dict) -> float    # returns end-to-end latency in ms

Phases:
  1. Fire    — fires `target_rps × duration_seconds` requests with arrival
               times spaced by `random.expovariate(target_rps)` (Poisson).
  2. Collect — awaits all in-flight tasks and re-sorts results by fire order
               so latencies align with their original arrival times.

Public functions a request_builder may call from `custom_run`:

    await engine.run_load(create_payload_fn, make_client_fn,
                          target_rps, duration_seconds, num_clients,
                          log_every, num_warmup, skip_first_n) -> dict

    engine.run_all_tests(config, request_builder) -> list

    engine.run_startup_probe(request_builder, config)
"""

import asyncio
import json
import random
import time

import numpy as np


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def _latency_metrics(latencies: list) -> dict:
    if not latencies:
        return {'mean_ms': 0.0, 'p50_ms': 0.0, 'p90_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0}
    arr = np.array(latencies)
    return {
        'mean_ms': float(np.mean(arr)),
        'p50_ms':  float(np.percentile(arr, 50)),
        'p90_ms':  float(np.percentile(arr, 90)),
        'p95_ms':  float(np.percentile(arr, 95)),
        'p99_ms':  float(np.percentile(arr, 99)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def _next_fire_time(target_rps: int) -> float:
    # Exponential inter-arrival time → Poisson arrival process.
    return random.expovariate(target_rps)


def _log_fire(elapsed, fired, total, in_flight):
    print(f'  {elapsed:6.2f}s | fired={fired}/{total} | in_flight≈{in_flight}')


def _log_collect(elapsed, completed, failures, total):
    print(f'  +{elapsed:6.2f}s | collected={completed + failures}/{total} | ok={completed} err={failures}')


# ---------------------------------------------------------------------------
# Startup probe
# ---------------------------------------------------------------------------

def run_startup_probe(request_builder, config: dict):
    """
    Fire one request before the load test to verify the endpoint is alive
    and let the operator inspect the exact payload being sent.

    If request_builder defines startup_probe(config), that is called instead.
    """
    payload = request_builder.create_payload(config)

    print('\n' + '=' * 80)
    print('  STARTUP PROBE')
    print('=' * 80)
    print('\n--- Payload ---')
    print(json.dumps(payload, indent=2, default=str))
    print()

    if hasattr(request_builder, 'startup_probe'):
        request_builder.startup_probe(config)
        print('=' * 80)
        return

    async def _probe():
        async with request_builder.make_client(config, 1) as invoke:
            return await invoke(payload)

    try:
        # _probe is an async function that fires a single request, asyncio.run is the engine necessary to
        # run an async function. It returns the latency for the request
        latency = asyncio.run(_probe())
        print(f'Endpoint responding ({latency:.0f} ms) — starting load test')
    except Exception as exc:
        print(f'Startup probe failed: {exc}')
        print('Aborting — fix the endpoint before running the load test.')
        raise SystemExit(1)
    print('=' * 80)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

async def _run_warmup(invoke_fn, create_payload_fn, num_warmup: int):
    """
    Fire `num_warmup` requests concurrently to prime the endpoint and any
    server-side connection pools. Errors are aggregated and printed by
    distinct message so latent server-side cold-start bugs are easy to spot.
    """
    if num_warmup <= 0:
        print('Skipping warmup (num_warmup_requests = 0)')
        return

    print(f'\nWarming up with {num_warmup} requests...')
    successful = failed = 0
    error_counts: dict = {}
    tasks = [asyncio.create_task(invoke_fn(create_payload_fn())) for _ in range(num_warmup)]

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        try:
            await coro
            successful += 1
        except Exception as e:
            failed += 1
            key = str(e)
            error_counts[key] = error_counts.get(key, 0) + 1
        if (i + 1) % max(1, num_warmup // 5) == 0:
            print(f'   Warmup progress: {i + 1}/{num_warmup}')

    print(f'   Warmup complete: {successful}/{num_warmup} successful')
    if failed:
        print(f'   {failed} warmup requests failed:')
        for msg, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f'     [{count}x] {msg}')
    print('   Waiting 2s before load test...\n')
    await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Core load test
# ---------------------------------------------------------------------------

async def run_load(create_payload_fn, make_client_fn,
                   target_rps: int, duration_seconds: int, num_clients: int,
                   log_every: float = 1.0, num_warmup: int = 0,
                   skip_first_n: int = 0) -> dict:
    """
    Two-phase Poisson load test. The transport is fully abstracted behind
    `make_client_fn` — the engine never touches aiohttp / aioboto3 / any
    other transport library directly.

    skip_first_n trims the first N requests from latency stats (cold-start
    exclusion without a separate warmup phase).
    """
    assert target_rps > 0 and duration_seconds > 0 and num_clients > 0
    total_requests = int(target_rps * duration_seconds)

    print(f'Target RPS: {target_rps}  |  Duration: {duration_seconds}s  |  Total: {total_requests} reqs')
    print(f'Async clients: {num_clients}')
    if skip_first_n > 0:
        print(f'First {skip_first_n} requests excluded from latency analysis')
    print()

    async with make_client_fn() as invoke_fn:

        if num_warmup > 0:
            await _run_warmup(invoke_fn, create_payload_fn, num_warmup)

        tasks = []
        fired = completed = failures = 0
        all_request_times = []  # wall-clock fire times — useful for plotting

        loop = asyncio.get_running_loop()
        start = loop.time()
        test_start_absolute = time.time()
        next_log = start + log_every
        next_fire = start

        print('Fire phase starting...')

        async def _indexed(idx: int):
            # Wrap invoke to preserve original fire order for post-sort.
            lat = await invoke_fn(create_payload_fn())
            return idx, lat

        # --- Phase 1: Fire ---
        for i in range(total_requests):
            now = loop.time()
            # Sleep only if ahead of schedule; reset next_fire to now otherwise
            # so we don't accumulate a backlog of debt.
            if now < next_fire:
                await asyncio.sleep(next_fire - now)
            else:
                next_fire = now
            next_fire += _next_fire_time(target_rps)

            all_request_times.append(time.time())
            tasks.append(asyncio.create_task(_indexed(i)))
            fired += 1

            now = loop.time()
            if now >= next_log:
                _log_fire(now - start, fired, total_requests, fired - completed - failures)
                next_log += log_every

        fire_done = loop.time()
        fire_duration = fire_done - start
        achieved_fire_rps = fired / fire_duration if fire_duration > 0 else 0
        print(f'Fire complete in {fire_duration:.2f}s (fire rate ≈ {achieved_fire_rps:.1f} rps)')

        # --- Phase 2: Collect ---
        print('Collect phase: waiting for responses...')
        next_log = loop.time() + log_every
        ordered_results = []

        # timeout gives extra headroom beyond the test duration for slow responses.
        for coro in asyncio.as_completed(tasks, timeout=duration_seconds + 300):
            try:
                idx, lat = await coro
                ordered_results.append((idx, lat, all_request_times[idx]))
                completed += 1
            except Exception as e:
                if failures == 0:
                    print(f'Error: {e}')
                failures += 1

            now = loop.time()
            if now >= next_log:
                _log_collect(now - fire_done, completed, failures, total_requests)
                next_log += log_every

    total_elapsed = loop.time() - start

    # Re-sort by original fire index so request_times aligns with latencies
    # for time-series plotting (as_completed returns in completion order).
    ordered_results.sort(key=lambda x: x[0])
    latencies = [r[1] for r in ordered_results]
    request_times = [r[2] for r in ordered_results]
    analysis_latencies = latencies[skip_first_n:]

    achieved_total_rps = (completed + failures) / fire_duration if fire_duration > 0 else 0

    summary = {
        'target_rps': target_rps,
        'duration_s': duration_seconds,
        'num_clients': num_clients,
        'fired': fired,
        'completed': completed,
        'failures': failures,
        'skip_first_n_requests': skip_first_n,
        'achieved_fire_rps': achieved_fire_rps,
        'achieved_total_rps': achieved_total_rps,
        'total_elapsed_s': total_elapsed,
        'latencies_ms': latencies,
        'request_times': request_times,
        'analysis_latencies_ms': analysis_latencies,
        'test_start_time': test_start_absolute,
    }

    output_file_path = "benchmark_results.txt"

    if analysis_latencies:
        metrics = _latency_metrics(analysis_latencies)
        summary.update(metrics)

        lines = [
            f'Done. Completed={completed}/{total_requests}, Failures={failures}, Achieved ~{achieved_total_rps:.1f} rps over {total_elapsed:.2f}s\n',
            f'Latency (ms): mean={metrics["mean_ms"]:.1f} p50={metrics["p50_ms"]:.1f} p90={metrics["p90_ms"]:.1f} p95={metrics["p95_ms"]:.1f} p99={metrics["p99_ms"]:.1f} min={metrics["min"]:.1f} max={metrics["max"]:.1f}\n'
        ]

        if skip_first_n > 0:
            lines.append(f'   (computed on {len(analysis_latencies)}/{completed} requests, first {skip_first_n} discarded)\n')
    else:
        lines = [f'No successful responses. Failures={failures}\n']

    print("".join(lines), end="")

    with open(output_file_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

    return summary


# ---------------------------------------------------------------------------
# Standard orchestrator
# ---------------------------------------------------------------------------

def run_all_tests(config: dict, request_builder) -> list:
    """
    Default orchestrator: loops variants × RPS values, calls `run_load` for
    each combination, prints an aggregate summary.

    Routes that need non-standard structure (matrix sweeps, multiple
    categories, …) implement `custom_run(config) -> list` on the
    request_builder; this function defers entirely to it when present.
    """
    if hasattr(request_builder, 'custom_run'):
        return request_builder.custom_run(config)

    lt = config['load_test']
    target_rps_value = lt['target_rps']
    target_rps_list = target_rps_value if isinstance(target_rps_value, list) else [target_rps_value]
    num_warmup = lt.get('num_warmup_requests', 20)
    cooldown = lt.get('cooldown_seconds', 0)
    skip_first_n = lt.get('skip_first_n_requests', lt.get('discard_first_n', 0))

    variants = (
        request_builder.get_variants(config)
        if hasattr(request_builder, 'get_variants')
        else [None]
    )

    all_summaries = []
    test_counter = 0
    total_tests = len(variants) * len(target_rps_list)

    for variant in variants:
        print('\n' + '=' * 80)
        print('  INITIALIZING TEST DATA')
        print('=' * 80)
        request_builder.initialize(config, variant)

        if hasattr(request_builder, 'post_init_report'):
            request_builder.post_init_report(config)

        def create_payload_fn():
            return request_builder.create_payload(config)

        def make_client_fn():
            return request_builder.make_client(config, lt['num_clients'])

        run_startup_probe(request_builder, config)

        for target_rps in target_rps_list:
            if test_counter > 0 and cooldown > 0:
                print(f'\nCooldown: waiting {cooldown}s...')
                time.sleep(cooldown)

            test_counter += 1
            desc = f'{target_rps} RPS'
            if hasattr(request_builder, 'variant_label') and variant is not None:
                desc += f' + {request_builder.variant_label(variant)}'
            if total_tests > 1:
                desc += f' ({test_counter}/{total_tests})'

            print('\n' + '=' * 80)
            print(f'  TESTING: {desc}')
            print('=' * 80)

            summary = asyncio.run(run_load(
                create_payload_fn=create_payload_fn,
                make_client_fn=make_client_fn,
                target_rps=target_rps,
                duration_seconds=lt['duration_seconds'],
                num_clients=lt['num_clients'],
                log_every=lt.get('log_every', 1.0),
                num_warmup=num_warmup,
                skip_first_n=skip_first_n,
            ))

            if variant is not None:
                label = (request_builder.variant_label(variant)
                         if hasattr(request_builder, 'variant_label') else str(variant))
                summary['variant'] = label

            if hasattr(request_builder, 'extra_summary_fields'):
                summary.update(request_builder.extra_summary_fields(config))

            all_summaries.append(summary)

    return all_summaries