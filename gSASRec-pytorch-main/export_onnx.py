import torch
import torch.nn as nn
from argparse import ArgumentParser
from utils import load_config, build_model, get_device


# model wrapper to export directly the predictions
class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    # forward function exported and used in Rust
    def forward(self, input_seq, limit):
        indices, values = self.original_model.get_predictions(input_seq, limit)
        return indices, values

# function written to export the model, isolated from the training
def export_model(saved_model_path = "gsasrec-ml1m-step_4465-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.019461369383357036.pt"):
    saved_model_path = "models/" + saved_model_path

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config_ml1m.py')
    args = parser.parse_args()
    config = load_config(args.config)

    device = get_device()
    model = build_model(config)

    model.load_state_dict(torch.load(saved_model_path, map_location=device))

    model.to("cpu")
    model.eval()
    seq_length = config.sequence_length

    # wrapped model
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    dummy_input = torch.randint(0, 1000, (1, seq_length), dtype=torch.long)
    dummy_k = torch.tensor([10], dtype=torch.long)

    onnx_file_name = saved_model_path.replace(".pt", ".onnx")
    torch.onnx.export(
        wrapped_model,
        (dummy_input, dummy_k,),
        onnx_file_name,
        input_names=['input_seq', 'limit'],
        output_names=['indices', 'values'],

        dynamic_axes={
            'input_seq': {0: 'batch_size'}
        }
    )

    print("Model exported with success.")
