import time
import numpy as np
import os
from contextlib import contextmanager


WARMUP_REQS = 30


@contextmanager
def track_model_latency(model_name: str):
    """Measure time and save the latencies in latencies.csv"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # all the workers write in the shared file
        with open("latencies.csv", "a") as f:
            f.write(f"{model_name},{latency_ms}\n")


def get_percentiles():
    """Read latencies.csv, discard the warmup rows and do the percentiles, then clean the file"""
    if not os.path.exists("latencies.csv"):
        return {"error": "No registered data. File not found."}

    # lists for all the latencies
    raw_latencies = {"pytorch": [], "onnx": []}

    with open("latencies.csv", "r") as f:
        for line in f:
            try:
                name, val = line.strip().split(",")
                if name in raw_latencies:
                    raw_latencies[name].append(float(val))
            except ValueError:
                continue

    results = {}

    for model_name, latencies_list in raw_latencies.items():
        total_recorded = len(latencies_list)

        # discard the warmup requests
        valid_latencies = latencies_list[WARMUP_REQS:]

        if not valid_latencies:
            # if the total requests are less than the warmup requests
            results[model_name] = {
                "error": f"Insufficient data. {total_recorded} requests registered, needed > {WARMUP_REQS}."
            }
            continue
        arr =np.array(valid_latencies)
        results[model_name] = {
            'mean_ms': float(np.mean(arr)),
            'p50_ms':  float(np.percentile(arr, 50)),
            'p90_ms':  float(np.percentile(arr, 90)),
            'p95_ms':  float(np.percentile(arr, 95)),
            'p99_ms':  float(np.percentile(arr, 99)),
            "total_requests_recorded": total_recorded,
            "warmup_skipped": WARMUP_REQS,
            "valid_requests_measured": len(valid_latencies)
        }

    # clean the file
    open("latencies.csv", "w").close()

    return results