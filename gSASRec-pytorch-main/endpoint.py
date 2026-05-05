import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from argparse import ArgumentParser
import uvicorn

from utils import build_model, get_device, load_config

MAX_LENGTH = 200
PADDING_VALUE = 0

# default server config
SERVER_CONFIG = {
    "config_path": "config_ml1m.py",
    "checkpoint_path": "models/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt"
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
        return  # Stop loading

    try:
        # paths from global dict
        config = load_config(SERVER_CONFIG["config_path"])
        device = get_device()

        print(f"Using device: {device}")

        model = build_model(config)
        model = model.to(device)
        model.load_state_dict(torch.load(SERVER_CONFIG["checkpoint_path"], map_location=device))
        model.eval()

        server_memory["model"] = model
        server_memory["device"] = device
        print("GSASRec model successfully loaded and ready to respond!")

    except Exception as e:
        print(f"Critical error during model loading: {e}")

    # the server starts listening thanks to yield
    yield

    # Clean shutdown at the end
    print("Shutting down the server... Clearing memory.")
    server_memory.clear()


app = FastAPI(lifespan=lifespan)

# health check functions
# if it is called the root
@app.api_route("/", methods=["GET", "POST", "OPTIONS"])
async def health_check_root():
    return {"status": "ok", "message": "GSASRec server online!"}

# if it is called the model but using get and not post
@app.api_route("/get_embeddings", methods=["GET", "OPTIONS"])
async def health_check_endpoint():
    return {"status": "ok"}

@app.post("/get_embeddings", response_model=EmbeddingsResponse)
async def get_embeddings(request: EmbeddingsRequest):
    # security check: verify that the model was correctly loaded at startup
    if "model" not in server_memory:
        raise HTTPException(
            status_code=500,
            detail="The model was not loaded correctly. Please check your terminal for errors about missing files!"
        )

    model = server_memory["model"]
    device = server_memory["device"]

    try:
        padded_batch = []

        # we loop through every user sequence sent by the client
        for sequence in request.batch_sequences:
            # if the sequence is longer than MAX_LENGTH, we keep only the most recent items
            if len(sequence) > MAX_LENGTH:
                sequence = sequence[-MAX_LENGTH:]

            # calculate how many padding spots we need to fill
            padding_length = MAX_LENGTH - len(sequence)

            # create the padded sequence
            padded_sequence = ([PADDING_VALUE] * padding_length) + sequence
            padded_batch.append(padded_sequence)

        input_tensor = torch.tensor(padded_batch, dtype=torch.long)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            seq_emb, attentions = model(input_tensor)
            final_embeddings_list = seq_emb.tolist()

        # return the formatted result
        return EmbeddingsResponse(embeddings=final_embeddings_list)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the tensor: {str(e)}")

#########################################
# to try it, use
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
    parser.add_argument('--config', type=str, default='config_optuna.py')
    parser.add_argument('--checkpoint', type=str,
                        default="models/gsasrec-ml1m-step_9310-t_0.5-negs_256-emb_256-dropout_0.16519583830077267-metric_0.1349321142424068.pt")

    args = parser.parse_args()

    # save the arguments globally so the async function can read them
    SERVER_CONFIG["config_path"] = args.config
    SERVER_CONFIG["checkpoint_path"] = args.checkpoint

    uvicorn.run(app, host="0.0.0.0", port=8081)