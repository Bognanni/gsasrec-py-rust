# start profiling before the test with curl -X POST http://localhost:8080/debug/start_profiling
# end profiling after the test with curl -X POST http://localhost:8080/debug/stop_profiling

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from argparse import ArgumentParser
import uvicorn
import gsasrec_rust
import orjson
from fastapi import Response

from utils import build_model, get_device, load_config
from latency_test import get_percentiles, track_model_latency

import cProfile
import pstats
import io

MAX_LENGTH = 200
PADDING_VALUE = 0


def pad_batch(sequences):
    clipped = [seq[-MAX_LENGTH:] for seq in sequences]
    target = max((len(s) for s in clipped), default=1)
    return [([PADDING_VALUE] * (target - len(s))) + s for s in clipped]


# --- Dynamic batching configuration ---
# MAX_BATCH_SIZE: hard cap on how many requests get merged into a single
#   forward pass. Keep it a multiple of typical per-request batch sizes
#   (e.g. 16) so GPU utilization stays high without growing tensors unbounded.
# MAX_WAIT_SECONDS: how long the batcher waits for more requests to arrive
#   before firing whatever it has collected so far. This bounds the extra
#   latency a single request can incur waiting to be batched with others.
MAX_BATCH_SIZE = int(os.environ.get("GSASREC_MAX_BATCH_SIZE", "1"))
MAX_WAIT_SECONDS = float(os.environ.get("GSASREC_MAX_WAIT_MS", "1")) / 1000.0


@dataclass
class _BatchItem:
    sequences: list  # list[list[int]] already padded to MAX_LENGTH
    future: "asyncio.Future"  # resolved with the per-item embeddings (np.ndarray) or an exception


class DynamicBatcher:
    """
    Accumulates individual requests into a queue, then merges everything
    that arrived within a short window (or up to MAX_BATCH_SIZE items) into
    a single call to `infer_fn`, which receives one big concatenated batch
    and must return one array indexable the same way (axis 0 = item).

    One instance per engine: each engine gets its own queue and its own
    background worker task, all driven by the single uvicorn event loop
    (this assumes --workers 1; with multiple worker processes each process
    would have its own independent batcher, which is fine since each holds
    its own copy of the model anyway).
    """

    def __init__(self, name: str, infer_fn: Callable[[list], Any],
                 max_batch_size: int = MAX_BATCH_SIZE,
                 max_wait_seconds: float = MAX_WAIT_SECONDS):
        self.name = name
        self.infer_fn = infer_fn
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self.queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, sequences: list) -> Any:
        """Called from the request handler. Awaits until this item's slice
        of a (possibly merged) batch has been computed."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(_BatchItem(sequences=sequences, future=future))
        return await future

    async def _worker_loop(self):
        while True:
            try:
                first_item = await self.queue.get()
            except asyncio.CancelledError:
                return

            batch_items = [first_item]
            deadline = asyncio.get_running_loop().time() + self.max_wait_seconds

            # Keep absorbing items already sitting in the queue (no wait),
            # then absorb newly arriving ones until the deadline or the
            # size cap, whichever comes first.
            while len(batch_items) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break

            await self._run_batch(batch_items)

    def _sync_infer_and_serialize(self, merged_sequences: list, item_sizes: list[int]) -> list[bytes]:
        """
        It runs entirely in a separate worker thread (CPU/GPU bound).
        It performs the forward pass and serializes the individual responses into raw JSON (bytes).
        """
        raw_result = self.infer_fn(merged_sequences)

        serialized_results = []
        offset = 0
        for size in item_sizes:
            user_slice = raw_result[offset: offset + size]

            # heavyweight serialization of the numpy array into JSON bytes, using orjson for speed
            json_bytes = orjson.dumps(
                {"embeddings": user_slice},
                option=orjson.OPT_SERIALIZE_NUMPY
            )
            serialized_results.append(json_bytes)
            offset += size

        return serialized_results

    async def _run_batch(self, batch_items: list):
        merged_sequences = []
        item_sizes = []
        for item in batch_items:
            merged_sequences.extend(item.sequences)
            item_sizes.append(len(item.sequences))

        print(f"[{self.name}] batch: {len(batch_items)} requests merged -> {len(merged_sequences)} rows")

        try:
            # inference and serialization together in a separate thread to avoid blocking the event loop
            serialized_payloads = await asyncio.to_thread(
                self._sync_infer_and_serialize,
                merged_sequences,
                item_sizes
            )
        except Exception as exc:
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        for item, payload_bytes in zip(batch_items, serialized_payloads):
            if not item.future.done():
                item.future.set_result(payload_bytes)


SERVER_CONFIG = {
    "config_path": os.environ.get("GSASREC_CONFIG", "config_ml1m.py"),
    "checkpoint_path": os.environ.get("GSASREC_CHECKPOINT",
                                      "pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt"),
    "onnx_model_path": os.environ.get("GSASREC_ONNX",
                                      "pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962_wrapped_embeddings.onnx"),
    "safetensors_path": os.environ.get("GSASREC_CANDLE", "pre_trained/model.safetensors"),
    "device": os.environ.get("GSASREC_DEVICE", "cuda"),
    "engine": os.environ.get("GSASREC_ENGINE", "pytorch")
}


class EmbeddingsRequest(BaseModel):
    batch_sequences: list[list[int]]


server_memory = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting the server... Preparing the GSASRec model.")

    if not os.path.exists(SERVER_CONFIG["config_path"]):
        print(f"CRITICAL ERROR: Cannot find the config file at: '{SERVER_CONFIG['config_path']}'")
        yield
        return
    if not os.path.exists(SERVER_CONFIG["checkpoint_path"]):
        print(f"CRITICAL ERROR: Cannot find the checkpoint file at: '{SERVER_CONFIG['checkpoint_path']}'")
        yield
        return
    if not os.path.exists(SERVER_CONFIG["onnx_model_path"]):
        print(f"CRITICAL ERROR: Cannot find the .onnx file at: '{SERVER_CONFIG['onnx_model_path']}'")
        yield
        return
    if not os.path.exists(SERVER_CONFIG["safetensors_path"]):
        print(f"CRITICAL ERROR: Cannot find the .safetensors file at: '{SERVER_CONFIG['safetensors_path']}'")
        yield
        return

    try:
        device_str = SERVER_CONFIG["device"]
        server_memory["device"] = device_str
        engine_choice = SERVER_CONFIG["engine"]

        if device_str == "cuda" and not torch.cuda.is_available():
            print("WARNING: 'cuda' requested but no GPU found. Falling back to 'cpu'.")
            device_str = "cpu"
            SERVER_CONFIG["device"] = device_str

        device = torch.device(device_str)
        print(f"Using device: {device} | Selected Engine: {engine_choice}")

        # --- PYTORCH ---
        if engine_choice in ["pytorch", "all"]:
            config = load_config(SERVER_CONFIG["config_path"])
            model = build_model(config)
            model = model.to(device)
            model.load_state_dict(torch.load(SERVER_CONFIG["checkpoint_path"], map_location=device))
            model.eval()

            # limit the internal PyTorch threads
            # if device_str == "cpu":
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

            server_memory["pytorch_model"] = model
            print("GSASRec PyTorch model loaded!")

            def _pytorch_infer(merged_sequences):
                return inference_pytorch_sync(model, device, merged_sequences)

            server_memory["pytorch_batcher"] = DynamicBatcher("pytorch", _pytorch_infer)
            server_memory["pytorch_batcher"].start()

        # --- ONNX PYTHON ---
        if engine_choice in ["onnx_python", "all"]:
            import onnxruntime as ort
            onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device_str == "cuda" else [
                'CPUExecutionProvider']

            # limit the number of threads to 1 to avoid oversubscription
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            onnx_session = ort.InferenceSession(SERVER_CONFIG["onnx_model_path"], sess_options=opts,
                                                providers=onnx_providers)
            server_memory["onnx_model"] = onnx_session
            print("GSASRec ONNX (Python) model loaded!")

            def _onnx_infer(merged_sequences):
                return inference_onnx(onnx_session, merged_sequences)

            server_memory["onnx_python_batcher"] = DynamicBatcher("onnx_python", _onnx_infer)
            server_memory["onnx_python_batcher"].start()

        # --- ONNX RUST ---
        if engine_choice in ["onnx_rust", "all"]:
            rust_session = gsasrec_rust.Recommender(SERVER_CONFIG["onnx_model_path"], device_str)
            server_memory["rust_model"] = rust_session
            print("GSASRec ONNX (Rust) model loaded!")

            def _onnx_rust_infer(merged_sequences):
                return inference_onnx_rust(rust_session, merged_sequences)

            server_memory["onnx_rust_batcher"] = DynamicBatcher("onnx_rust", _onnx_rust_infer)
            server_memory["onnx_rust_batcher"].start()

        # --- CANDLE RUST ---
        if engine_choice in ["candle_rust", "all"]:
            candle_session = gsasrec_rust.CandleRecommender(SERVER_CONFIG["safetensors_path"], device_str)
            server_memory["candle_rust_model"] = candle_session
            print("GSASRec Candle (Rust) model loaded!")

            def _candle_infer(merged_sequences):
                return inference_candle(candle_session, merged_sequences)

            server_memory["candle_rust_batcher"] = DynamicBatcher("candle_rust", _candle_infer)
            server_memory["candle_rust_batcher"].start()

    except Exception as e:
        print(f"Critical error during models' loading: {e}")

    yield
    print("Shutting down the server... Clearing memory.")
    for key in ("pytorch_batcher", "onnx_python_batcher", "onnx_rust_batcher", "candle_rust_batcher"):
        batcher = server_memory.get(key)
        if batcher is not None:
            await batcher.stop()
    server_memory.clear()


app = FastAPI(lifespan=lifespan)


@app.api_route("/", methods=["GET", "POST", "OPTIONS"])
async def health_check_root():
    return {"status": "ok", "message": "GSASRec server online!"}


@app.api_route("/get_embeddings", methods=["GET", "OPTIONS"])
async def health_check_endpoint():
    return {"status": "ok"}


profiler_state = {"profiler": None, "active": False}


@app.post("/debug/start_profiling")
async def start_profiling():
    if profiler_state["active"]:
        return {"status": "already running"}
    profiler_state["profiler"] = cProfile.Profile()
    profiler_state["profiler"].enable()
    profiler_state["active"] = True
    return {"status": "profiling started"}


@app.post("/debug/stop_profiling")
async def stop_profiling():
    if not profiler_state["active"]:
        return {"status": "not running"}
    profiler_state["profiler"].disable()
    profiler_state["active"] = False

    s = io.StringIO()
    ps = pstats.Stats(profiler_state["profiler"], stream=s).sort_stats('cumulative')
    ps.print_stats(40)

    return {"profile": s.getvalue()}


def inference_pytorch_sync(model, device, padded_batch):
    input_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)
    with torch.no_grad():
        seq_emb, _ = model(input_tensor)
        seq_emb = seq_emb[:, -1, :]
        if device.type == "cuda":
            torch.cuda.synchronize()
    return seq_emb.cpu().numpy()


@app.post("/get_embeddings/pytorch")
async def get_embeddings_checkpoint(request: EmbeddingsRequest):
    if "pytorch_batcher" not in server_memory:
        raise HTTPException(status_code=500, detail="PyTorch model missing!")

    batcher: DynamicBatcher = server_memory["pytorch_batcher"]

    try:
        padded_batch = pad_batch(request.batch_sequences)

        final_payload_bytes = await batcher.submit(padded_batch)

        return Response(
            content=final_payload_bytes,
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


def inference_onnx(session, padded_batch):
    input_array = np.array(padded_batch, dtype=np.int64)

    outputs = session.run(["embedded"], {"input_seq": input_array})
    outputs = outputs[0][:, -1, :]

    return outputs


@app.post("/get_embeddings/onnx_python")
async def get_embeddings_onnx(request: EmbeddingsRequest):
    if "onnx_python_batcher" not in server_memory:
        raise HTTPException(status_code=500, detail="ONNX model missing!")

    batcher: DynamicBatcher = server_memory["onnx_python_batcher"]

    try:
        padded_batch = pad_batch(request.batch_sequences)
        final_embeddings_array = await batcher.submit(padded_batch)

        return Response(
            content=final_embeddings_array,
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error ONNX: {str(e)}")


def inference_onnx_rust(rust_model, padded_batch):
    flat_embeddings = rust_model.get_embeddings(padded_batch)

    batch_size = len(padded_batch)
    arr = np.array(flat_embeddings, dtype=np.float32)
    reshaped_batch = arr.reshape(batch_size, -1)

    return reshaped_batch


@app.post("/get_embeddings/onnx_rust")
async def get_embeddings_onnx_rust(request: EmbeddingsRequest):
    if "onnx_rust_batcher" not in server_memory:
        raise HTTPException(status_code=500, detail="Rust ONNX model missing!")

    batcher: DynamicBatcher = server_memory["onnx_rust_batcher"]

    try:
        padded_batch = pad_batch(request.batch_sequences)
        reshaped_batch = await batcher.submit(padded_batch)

        return Response(
            content=reshaped_batch,
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error Rust ONNX model: {str(e)}")


def inference_candle(candle_model, padded_batch):
    flat_embeddings = candle_model.get_embeddings(padded_batch)

    batch_size = len(padded_batch)
    arr = np.array(flat_embeddings, dtype=np.float32)
    reshaped_batch = arr.reshape(batch_size, -1)

    return reshaped_batch


@app.post("/get_embeddings/candle_rust")
async def get_embeddings_candle_rust(request: EmbeddingsRequest):
    if "candle_rust_batcher" not in server_memory:
        raise HTTPException(status_code=500, detail="Rust Candle model missing!")

    batcher: DynamicBatcher = server_memory["candle_rust_batcher"]

    try:
        padded_batch = pad_batch(request.batch_sequences)
        reshaped_batch = await batcher.submit(padded_batch)

        return Response(
            content=reshaped_batch,
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error Rust Candle model: {str(e)}")


@app.get("/metrics/model-latency")
def model_latency_metrics():
    return get_percentiles()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config_ml1m.py')
    parser.add_argument('--checkpoint', type=str,
                        default="pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt")
    parser.add_argument("--onnx", type=str,
                        default="pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962_wrapped_embeddings.onnx")
    parser.add_argument("--candle", type=str, default="pre_trained/model.safetensors")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--engine", type=str, choices=["pytorch", "onnx_python", "onnx_rust", "candle_rust", "all"],
                        default="pytorch")
    args = parser.parse_args()

    os.environ["GSASREC_CONFIG"] = args.config
    os.environ["GSASREC_CHECKPOINT"] = args.checkpoint
    os.environ["GSASREC_ONNX"] = args.onnx
    os.environ["GSASREC_CANDLE"] = args.candle
    os.environ["GSASREC_DEVICE"] = args.device
    os.environ["GSASREC_ENGINE"] = args.engine

    uvicorn.run("endpoint:app", host="0.0.0.0", port=8080, workers=args.workers)