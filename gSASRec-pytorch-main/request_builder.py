"""
request_builder for /embed_sequence — local HTTP transport.

Same payload shape as the SageMaker version next door; only the transport
differs. Compare side-by-side: the only real change is `make_client`.
"""
import csv
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

from http_transport import make_http_client

_ITEM_IDS: list = []


def _load_csv(path: Path) -> list:
    if not path.exists():
        print(f'  {path.name} not found — falling back to synthetic UUIDs')
        return []
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                rows.append(row[0])
    print(f'  Loaded {len(rows):,} item IDs from {path.name}')
    return rows


def initialize(config: dict, variant=None) -> None:
    global _ITEM_IDS
    csv_path = config.get('request', {}).get('ids_csv')
    if csv_path:
        _ITEM_IDS = _load_csv((Path(__file__).parent / csv_path).resolve())
    else:
        _ITEM_IDS = []


def _sequence(length: int) -> list:
    """
    Generates a single flat list of integer item IDs.
    Example output: [15, 22, 108, 5, ...]
    """
    if _ITEM_IDS:
        # choose random items from your list, making sure they are integers
        return [int(random.choice(_ITEM_IDS)) for _ in range(length)]

    # if no IDs are provided, generate random integers
    return [random.randint(1, 3415) for _ in range(length)]


def create_payload(config: dict) -> dict:
    """
    Builds the JSON payload with multiple users in a single batch.
    """
    req = config.get('request', {})

    # take the sequence length from the config (how many real items the user interacted with)
    sequence_length = int(req.get('sequence_length', 40))
    # size of the batch that we can change if we want
    batch_size = 16

    # multiple sequences (one for each fake user)
    batch_of_users = []
    for _ in range(batch_size):
        user_sequence = _sequence(sequence_length)
        batch_of_users.append(user_sequence)

    # return the fully packed batch
    return {
        'batch_sequences': batch_of_users
    }


def get_headers(config: dict) -> dict:
    headers = {'Content-Type': 'application/json'}
    headers.update(config.get('headers', {}) or {})
    if token := os.environ.get('AUTH_TOKEN'):
        headers['Authorization'] = f'Bearer {token}'
    return headers


@asynccontextmanager
async def make_client(config: dict, num_clients: int):
    c = config['endpoint'].get('client', {})
    url = config['endpoint']['url'] + config['request']['path']
    async with make_http_client(
        url=url,
        get_headers_fn=lambda: get_headers(config),
        num_clients=num_clients,
        connect_timeout=c.get('connect_timeout', 3),
        read_timeout=c.get('read_timeout', 60),
    ) as invoke:
        yield invoke


def post_init_report(config: dict) -> None:
    req = config.get('request', {})
    print(f"  Endpoint         : {config['endpoint']['url']}")
    print(f"  Path             : {req.get('path', '/embed_sequence')}")
    print(f"  Sequence length  : {req.get('sequence_length', 40)}")
    print(f"  ID source        : {'CSV (' + str(len(_ITEM_IDS)) + ' ids)' if _ITEM_IDS else 'synthetic UUIDs'}")
