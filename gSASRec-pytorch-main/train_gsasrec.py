from argparse import ArgumentParser
import os
import torch
from utils import load_config, build_model, get_device
from dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from tqdm import tqdm
from eval_utils import evaluate
from torchinfo import summary
from export_onnx import export_model

models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
args = parser.parse_args()
config = load_config(args.config)

num_items = get_num_items(config.dataset_name) 
device = get_device()
model = build_model(config)

# get the dataloader for training, evaluation and test personalized
train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size,
                                         max_length=config.sequence_length, train_neg_per_positive=config.negs_per_pos)
val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)

optimiser = torch.optim.Adam(model.parameters())
batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

model = model.to(device)
# summary of the model, useful to do a visual sanity check
summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

print("Starting the training")

for epoch in range(config.max_epochs):
    model.train()   
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0
    # iter for a single batch
    for batch_idx in pbar:
        step += 1
        # take the positives and negatives specified in the train data loader and move them to the gpu
        positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
        # shift to have the input items
        model_input = positives[:, :-1]
        # forward of the transformer getting the representations (last_hidden_state) and the attention weights
        last_hidden_state, attentions = model(model_input)
        # shift to have the target items
        labels = positives[:, 1:]
        negatives = negatives[:, 1:, :]
        # concatenate the correct items with some random negatives (we are considering the IDs of the items)
        # unsqueeze is to have the same shape labels and negatives (at the start labels is [batch_size, sequence_length]
        # and negatives [batch_size, sequence_length, num_negatives], after the unsqueeze for labels we have
        # [batch_size, sequence_length, 1]
        pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
        # get the lookup table
        output_embeddings = model.get_output_embeddings()
        # get the embeddings of the items in pos_neg_concat using the lookup table
        pos_neg_embeddings = output_embeddings(pos_neg_concat)
        mask = (model_input != num_items + 1).float()
        # compute the dot product between the state of the user and the list of candidates (for each user 1 correct item
        # and N negative wrong items)
        logits = torch.einsum('bse, bsne -> bsn', last_hidden_state, pos_neg_embeddings)
        # ground truth, 1 for the correct item and 0 for the others, this for each user
        gt = torch.zeros_like(logits)
        gt[:, :, 0] = 1

        # the whole block alter the probability using the generalized Binary Cross Entropy Loss to mitigate the Popularity Bias
        # probability that an item is chosen as negative
        alpha = config.negs_per_pos / (num_items - 1)
        # import the parameter about temperature
        t = config.gbce_t
        # how much penalty consider, following the gSASRec paper
        beta = alpha * ((1 - 1/alpha)*t + 1/alpha)
        # divide the tensor in two, positives and negatives
        positive_logits = logits[:, :, 0:1].to(torch.float64) #use float64 to increase numerical stability
        negative_logits = logits[:,:,1:].to(torch.float64)
        eps = 1e-10
        # transform the positive logits using the sigmoid, forcing to stay between 0 and 1, but never exactly 0 or 1
        # thanks to .clamp()
        positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1-eps)
        # apply the penalty computing 1 / (probability^beta)
        positive_probs_adjusted = torch.clamp(positive_probs.pow(-beta), 1+eps, torch.finfo(torch.float64).max)
        # return to raw logits because the loss function used later take in input raw logits and not probabilities
        to_log = torch.clamp(torch.div(1.0, (positive_probs_adjusted  - 1)), eps, torch.finfo(torch.float64).max)
        positive_logits_transformed = to_log.log()
        # return to the initial shape for the logits, unifying positives and negatives
        logits = torch.cat([positive_logits_transformed, negative_logits], -1)

        # BCE only for the real items, ignoring the padding
        loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1)*mask
        # mean error
        loss = loss_per_element.sum() / mask.sum()
        # backpropagation and optimization
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss_sum += loss.item()
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1)}")

    # evaluate the model trained using the val data loader at the end of an epoch
    evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                 config.filter_rated, device=device)
    # save the results if they are the best and the model checkpoint
    print(f"Epoch {epoch} evaluation result: {evaluation_result}")
    if evaluation_result[config.val_metric] > best_metric:
        best_metric = evaluation_result[config.val_metric]
        model_name = f"models/gsasrec-{config.dataset_name}-step:{step}-t:{config.gbce_t}-negs:{config.negs_per_pos}-emb:{config.embedding_dim}-dropout:{config.dropout_rate}-metric:{best_metric}.pt".replace(':', '_')
        print(f"Saving new best model to {model_name}")
        if best_model_name is not None:
            os.remove(best_model_name)
        best_model_name = model_name
        steps_not_improved = 0
        torch.save(model.state_dict(), model_name)
    else:
        steps_not_improved += 1
        print(f"Validation metric did not improve for {steps_not_improved} steps")
        if steps_not_improved >= config.early_stopping_patience:
            print(f"Stopping training, best model was saved to {best_model_name}")
            break

# export the model using ONNX
export_model(best_model_name)