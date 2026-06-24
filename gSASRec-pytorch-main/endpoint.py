import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from argparse import ArgumentParser
import uvicorn
import gsasrec_rust

from utils import build_model, get_device, load_config
from latency_test import get_percentiles, track_model_latency

MAX_LENGTH = 200
PADDING_VALUE = 0

SERVER_CONFIG = {
    "config_path": os.environ.get("GSASREC_CONFIG", "config_ml1m.py"),
    "checkpoint_path": os.environ.get("GSASREC_CHECKPOINT",
                                      "pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt"),
    "onnx_model_path": os.environ.get("GSASREC_ONNX",
                                      "pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.onnx"),
    "safetensors_path": os.environ.get("GSASREC_CANDLE", "pre_trained/model.safetensors"),
    "device": os.environ.get("GSASREC_DEVICE", "cuda"),
    "engine": os.environ.get("GSASREC_ENGINE", "pytorch")
}


class EmbeddingsRequest(BaseModel):
    batch_sequences: list[list[int]]


class EmbeddingsResponse(BaseModel):
    embeddings: list[list[list[float]]]


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
            if device_str == "cpu":
                torch.set_num_threads(1)

            server_memory["pytorch_model"] = model
            print("GSASRec PyTorch model loaded!")

        # --- ONNX PYTHON ---
        if engine_choice in ["onnx_python", "all"]:
            import onnxruntime as ort
            onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device_str == "cuda" else [
                'CPUExecutionProvider']

            # limit the number of threads to 1 to avoid oversubscription
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1

            onnx_session = ort.InferenceSession(SERVER_CONFIG["onnx_model_path"], sess_options=opts,
                                                providers=onnx_providers)
            server_memory["onnx_model"] = onnx_session
            print("GSASRec ONNX (Python) model loaded!")

        # --- ONNX RUST ---
        if engine_choice in ["onnx_rust", "all"]:
            rust_session = gsasrec_rust.Recommender(SERVER_CONFIG["onnx_model_path"], device_str)
            server_memory["rust_model"] = rust_session
            print("GSASRec ONNX (Rust) model loaded!")

        # --- CANDLE RUST ---
        if engine_choice in ["candle_rust", "all"]:
            candle_session = gsasrec_rust.CandleRecommender(SERVER_CONFIG["safetensors_path"], device_str)
            server_memory["candle_rust_model"] = candle_session
            print("GSASRec Candle (Rust) model loaded!")

    except Exception as e:
        print(f"Critical error during models' loading: {e}")

    yield
    print("Shutting down the server... Clearing memory.")
    server_memory.clear()


app = FastAPI(lifespan=lifespan)


@app.api_route("/", methods=["GET", "POST", "OPTIONS"])
async def health_check_root():
    return {"status": "ok", "message": "GSASRec server online!"}


@app.api_route("/get_embeddings", methods=["GET", "OPTIONS"])
async def health_check_endpoint():
    return {"status": "ok"}


@app.post("/get_embeddings/pytorch", response_model=EmbeddingsResponse)
def get_embeddings_checkpoint(request: EmbeddingsRequest):
    if "pytorch_model" not in server_memory:
        raise HTTPException(status_code=500, detail="PyTorch model missing!")

    model = server_memory["pytorch_model"]
    device = torch.device(server_memory["device"])

    try:
        padded_batch = [
            ([PADDING_VALUE] * (MAX_LENGTH - len(seq[-MAX_LENGTH:]))) + seq[-MAX_LENGTH:]
            for seq in request.batch_sequences
        ]
        input_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)

        with torch.no_grad():
            seq_emb, _ = model(input_tensor)
            final_embeddings_list = seq_emb.tolist()

        return EmbeddingsResponse(embeddings=final_embeddings_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.post("/get_embeddings/onnx_python", response_model=EmbeddingsResponse)
def get_embeddings_onnx(request: EmbeddingsRequest):
    if "onnx_model" not in server_memory:
        raise HTTPException(status_code=500, detail="ONNX model missing!")

    session = server_memory["onnx_model"]

    try:
        padded_batch = [
            ([PADDING_VALUE] * (MAX_LENGTH - len(seq[-MAX_LENGTH:]))) + seq[-MAX_LENGTH:]
            for seq in request.batch_sequences
        ]
        input_array = np.array(padded_batch, dtype=np.int64)

        outputs = session.run(["embedded"], {"input_seq": input_array})
        final_embeddings_list = outputs[0].tolist()

        return EmbeddingsResponse(embeddings=final_embeddings_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error ONNX: {str(e)}")


@app.post("/get_embeddings/onnx_rust", response_model=EmbeddingsResponse)
def get_embeddings_onnx_rust(request: EmbeddingsRequest):
    if "rust_model" not in server_memory:
        raise HTTPException(status_code=500, detail="Rust ONNX model missing!")

    rust_model = server_memory["rust_model"]

    try:
        padded_batch = [
            ([PADDING_VALUE] * (MAX_LENGTH - len(seq[-MAX_LENGTH:]))) + seq[-MAX_LENGTH:]
            for seq in request.batch_sequences
        ]
        flat_embeddings = rust_model.get_embeddings(padded_batch)
        batch_size = len(padded_batch)
        arr = np.array(flat_embeddings, dtype=np.float32)
        reshaped_batch = arr.reshape(batch_size, MAX_LENGTH, -1).tolist()

        return EmbeddingsResponse(embeddings=reshaped_batch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error Rust ONNX model: {str(e)}")


@app.post("/get_embeddings/candle_rust", response_model=EmbeddingsResponse)
def get_embeddings_candle_rust(request: EmbeddingsRequest):
    if "candle_rust_model" not in server_memory:
        raise HTTPException(status_code=500, detail="Rust Candle model missing!")

    candle_model = server_memory["candle_rust_model"]

    try:
        padded_batch = [
            ([PADDING_VALUE] * (MAX_LENGTH - len(seq[-MAX_LENGTH:]))) + seq[-MAX_LENGTH:]
            for seq in request.batch_sequences
        ]
        flat_embeddings = candle_model.get_embeddings(padded_batch)
        batch_size = len(padded_batch)
        arr = np.array(flat_embeddings, dtype=np.float32)
        reshaped_batch = arr.reshape(batch_size, MAX_LENGTH, -1).tolist()

        return EmbeddingsResponse(embeddings=reshaped_batch)
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
                        default="pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.onnx")
    parser.add_argument("--candle", type=str, default="pre_trained/model.safetensors")
    parser.add_argument("--workers", type=int, default=9, help="Number of uvicorn workers")
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