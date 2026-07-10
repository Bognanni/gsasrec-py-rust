import torch
import torch.nn as nn
from argparse import ArgumentParser
from utils import load_config, build_model, get_device


# model wrapper to export directly the predictions
class ModelWrapperPredictions(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    # forward function exported and used in Rust
    def forward(self, input_seq, limit):
        indices, values = self.original_model.get_predictions(input_seq, limit)
        return indices, values

# model wrapper to export directly the embeddings without the attentions
class ModelWrapperEmbeddings(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_seq):
        seq_emb, _ = self.model(input_seq)
        return seq_emb


# function written to export the model, isolated from the training
def export_model(saved_model_path="pre_trained/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.pt",
                 config="config_ml1m.py", embedded_results=True):
    model_config = load_config(config)

    device = get_device()
    model = build_model(model_config)

    model.load_state_dict(torch.load(saved_model_path, map_location=device))

    model.to("cpu")
    model.eval()
    seq_length = model_config.sequence_length

    dummy_input = torch.randint(0, 1000, (2, seq_length), dtype=torch.long)


    # wrapped model
    if not embedded_results:
        wrapped_model = ModelWrapperPredictions(model)
        wrapped_model.eval()
        dummy_k = torch.tensor([10], dtype=torch.long)
        onnx_file_name = saved_model_path.replace(".pt", "_wrapped_predictions.onnx")

        torch.onnx.export(
            wrapped_model,
            (dummy_input, dummy_k,),
            onnx_file_name,
            input_names=['input_seq', 'limit'],
            output_names=['indices', 'values'],
            dynamic_axes={
                # Anche qui l'input deve accettare sequenze di lunghezza variabile
                'input_seq': {0: 'batch_size', 1: 'sequence_length'},
                # Gli output devono poter scalare in base al batch e al parametro 'limit'
                'indices': {0: 'batch_size', 1: 'limit'},
                'values': {0: 'batch_size', 1: 'limit'}
            },
            opset_version=14
        )
    # wrapped model that returns only the embeddings without the attentions
    else:
        wrapped_model = ModelWrapperEmbeddings(model)
        wrapped_model.eval()
        onnx_file_name = saved_model_path.replace(".pt", "_wrapped_embeddings.onnx")

        torch.onnx.export(
            wrapped_model,
            (dummy_input,),
            onnx_file_name,
            input_names=['input_seq'],
            output_names=['embedded'],
            dynamic_axes={
                'input_seq': {
                    0: 'batch_size',
                    1: 'sequence_length'
                },
                'embedded': {
                    0: 'batch_size',
                    1: 'sequence_length'
                }
            },
            opset_version=14
        )

    print("Model exported with success.")

export_model()