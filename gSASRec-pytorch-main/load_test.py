"""Runner for /embed_sequence — local HTTP transport."""
import argparse
import sys
import requests
from pathlib import Path

import yaml

_ROUTE_DIR = Path(__file__).parent
_SHARE_DIR = _ROUTE_DIR.parent.parent.parent
sys.path.insert(0, str(_SHARE_DIR))
sys.path.insert(0, str(_ROUTE_DIR))

import request_builder
from engine import run_all_tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(_ROUTE_DIR / 'config.yaml'))
    parser.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.no_warmup:
        config['load_test']['num_warmup_requests'] = 0

    # run the tests with the internal logic
    run_all_tests(config, request_builder)

    # get the pure latencies
    # response = requests.get("http://localhost:8081/metrics/model-latency")
    # metrics_data = response.json()
    #
    # print(metrics_data)