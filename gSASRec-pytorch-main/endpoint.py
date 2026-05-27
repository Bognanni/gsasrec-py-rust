import os
import torch
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from argparse import ArgumentParser
import uvicorn
import gsasrec_rust

from utils import build_model, get_device, load_config
from latency_test import track_model_latency, get_percentiles

MAX_LENGTH = 200
PADDING_VALUE = 0

# default server config
SERVER_CONFIG = {
    "config_path": "config_ml1m.py",
    "checkpoint_path": "pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt",
    "onnx_model_path": "pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.onnx"
    }


# template for input data: a list of lists of integers
class EmbeddingsRequest(BaseModel):
    batch_sequences: list[list[int]]


# template for the response: final embeddings as a 3D list
class EmbeddingsResponse(BaseModel):
    embeddings: list[list[list[float]]]


# we use a dictionary to keep the model in memory
server_memory = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting the server... Preparing the GSASRec model.")

    # verify that the file exists
    if not os.path.exists(SERVER_CONFIG["config_path"]):
        print(f"CRITICAL ERROR: Cannot find the config file at: '{SERVER_CONFIG['config_path']}'")
        print("Please check your spelling or provide the correct --config path.")
        # yield before return because you can't use only return
        yield
        return

    if not os.path.exists(SERVER_CONFIG["checkpoint_path"]):
        print(f"CRITICAL ERROR: Cannot find the checkpoint file at: '{SERVER_CONFIG['checkpoint_path']}'")
        print("Please check your spelling or provide the correct --checkpoint path.")
        yield
        return

    if not os.path.exists(SERVER_CONFIG["onnx_model_path"]):
        print(f"CRITICAL ERROR: Cannot find the .onnx file at: '{SERVER_CONFIG['onnx_model_path']}'")
        print("Please check your spelling or provide the correct --onnx path.")
        yield
        return

    try:
        # paths from global dict
        config = load_config(SERVER_CONFIG["config_path"])
        device = get_device()

        print(f"Using device: {device}")

        model = build_model(config)
        model = model.to(device)
        model.load_state_dict(torch.load(SERVER_CONFIG["checkpoint_path"], map_location=device))
        model.eval()

        server_memory["pytorch_model"] = model
        server_memory["device"] = device
        print("GSASRec pytorch model successfully loaded and ready to answer!")

        # to put also the onnx model on the gpu
        onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        onnx_session = ort.InferenceSession(SERVER_CONFIG["onnx_model_path"], providers=onnx_providers)
        server_memory["onnx_model"] = onnx_session
        print("GSASRec ONNX model successfully loaded and ready to answer!")

        # add also the onnx model imported in rust

        rust_session = gsasrec_rust.Recommender(SERVER_CONFIG["onnx_model_path"])
        server_memory["rust_model"] = rust_session
        print("GSASRec RUST ONNX model successfully loaded and ready to answer!")

    except Exception as e:
        print(f"Critical error during models' loading: {e}")

    # the server starts listening thanks to yield
    yield

    # Clean shutdown at the end
    print("Shutting down the server... Clearing memory.")
    server_memory.clear()


app = FastAPI(lifespan=lifespan)

# middleware to eventually test the path
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     print(f"\n[MIDDLEWARE] Request arrived: {request.method} {request.url.path}")
#     response = await call_next(request)
#     print(f"[MIDDLEWARE] Response sent: Status {response.status_code}")
#     return response

# health check functions
# if it is called the root
@app.api_route("/", methods=["GET", "POST", "OPTIONS"])
async def health_check_root():
    return {"status": "ok", "message": "GSASRec server online!"}

# if it is called the model but using get and not post
@app.api_route("/get_embeddings", methods=["GET", "OPTIONS"])
async def health_check_endpoint():
    return {"status": "ok"}

@app.post("/get_embeddings/checkpoint", response_model=EmbeddingsResponse)
def get_embeddings_checkpoint(request: EmbeddingsRequest):
    # security check: verify that the model was correctly loaded at startup
    if "pytorch_model" not in server_memory:
        raise HTTPException(
            status_code=500,
            detail="The pytorch model was not loaded correctly. Please check your terminal for errors about missing files!"
        )

    model = server_memory["pytorch_model"]
    device = server_memory["device"]

    try:
        padded_batch = [
            ([PADDING_VALUE] * (MAX_LENGTH - len(seq[-MAX_LENGTH:]))) + seq[-MAX_LENGTH:]
            for seq in request.batch_sequences
        ]

        input_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)

        with torch.no_grad():
            with track_model_latency():
                seq_emb, _ = model(input_tensor)
            final_embeddings_list = seq_emb.tolist()

        # return the formatted result
        return EmbeddingsResponse(embeddings=final_embeddings_list)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the tensor: {str(e)}")

@app.post("/get_embeddings/onnx", response_model=EmbeddingsResponse)
def get_embeddings_onnx(request: EmbeddingsRequest):
    if "onnx_model" not in server_memory:
        raise HTTPException(
            status_code=500,
            detail="The ONNX model was not loaded correctly. Please check your terminal for errors about missing files!"
        )

    session = server_memory["onnx_model"]

    try:
        padded_batch = [
            ([PADDING_VALUE] * (MAX_LENGTH - len(seq[-MAX_LENGTH:]))) + seq[-MAX_LENGTH:]
            for seq in request.batch_sequences
        ]

        # numpy array for onnx
        input_array = np.array(padded_batch, dtype=np.int64)

        # run the ONNX model
        with track_model_latency():
            outputs = session.run(["embedded"], {"input_seq": input_array})
        final_embeddings_list = outputs[0].tolist()

        return EmbeddingsResponse(embeddings=final_embeddings_list)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error ONNX: {str(e)}")


@app.post("/get_embeddings/onnx_rust", response_model=EmbeddingsResponse)
def get_embeddings_onnx_rust(request: EmbeddingsRequest):
    if "rust_model" not in server_memory:
        raise HTTPException(
            status_code=500,
            detail="The Rust ONNX model was not loaded correctly. Ensure rust_engine is compiled and imported."
        )

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


@app.get("/metrics/model-latency")
def model_latency_metrics():
    """Returns the percentiles"""
    return get_percentiles()


#########################################
# to try it, use searching http://localhost:8081/docs on the browser
# {
#   "batch_sequences": [
#     [15, 22, 108],
#     [5, 10]
#   ]
# }
#########################################

if __name__ == "__main__":
    # parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config_ml1m.py')
    parser.add_argument('--checkpoint', type=str,
                        default="pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt")
    parser.add_argument("--onnx", type=str,
                        default="pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.onnx")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")

    args = parser.parse_args()

    # save the arguments globally so the async function can read them
    SERVER_CONFIG["config_path"] = args.config
    SERVER_CONFIG["checkpoint_path"] = args.checkpoint
    SERVER_CONFIG["onnx_model_path"] = args.onnx
    os.environ["ANYIO_NUM_THREADS"] = "32"

    uvicorn.run("endpoint:app", host="0.0.0.0", port=8081, workers=args.workers)