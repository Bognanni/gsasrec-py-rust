import json
import torch
from torch.utils.data import Dataset, DataLoader

# class to define a personalized dataset
class SequenceDataset(Dataset):
    def __init__(self, input_file, padding_value, output_file=None, max_length=200 ):
        # convert each row in a list of integer numbers
        with open(input_file, 'r') as f:
            # take each line, remove the \n using .split(), divide into a list of String using .strip()
            # apply the int function (to cast) to each value and then with list we have a list of integers
            self.inputs = [list(map(int, line.strip().split())) for line in f.readlines()]

        # same for the output file
        if output_file:
            with open(output_file, 'r') as f:
                self.outputs = [int(line.strip()) for line in f.readlines()]
        else:
            self.outputs = None

        self.max_length = max_length
        self.padding_value = padding_value

    # total number of sequences (users)
    def __len__(self):
        return len(self.inputs)

    # take the sequence for a single user (idx) and save in a set all the items (rated)
    def __getitem__(self, idx):
        inp = self.inputs[idx]
        # all the items consumed in all the history by the user (without order and without replication)
        rated = set(inp)
        # if the sequence is too long mantain only the last items, if is too short add padding
        if len(inp) > self.max_length:
            inp = inp[-self.max_length:]
        elif len(inp) < self.max_length:
            inp = [self.padding_value] * (self.max_length - len(inp)) + inp

        # convert into a tensor (this is the actual sequence passed to the model, with padding if necessary and with the
        # fixed length and repetition if any)
        inp_tensor = torch.tensor(inp, dtype=torch.long)

        # same for the output
        if self.outputs:
            out_tensor = torch.tensor(self.outputs[idx], dtype=torch.long)
            return inp_tensor, rated, out_tensor 

        return inp_tensor,

# take a list of single samples and stack them creating a batch tensor, then generate a matrix of random negatives
def collate_with_random_negatives(input_batch, pad_value, num_negatives):
    batch_cat = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    negatives = torch.randint(low=1, high=pad_value, size=(batch_cat.size(0), batch_cat.size(1), num_negatives))
    return [batch_cat, negatives]

# define a method to unify tensors and set, for validation and test, without generating negatives
# consider that input_batch is a list of tuples with 3 elements each [(tensor_seq_user, set_single_items_consumed,
# tensor_target)] -> user 1
def collate_val_test(input_batch):
    # stack the tensors into a single tensor (ex. 32 users with each sequence long 200, the result is [32, 200])
    input = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    # take the second element of input_batch (a set of the items already consumed from each user) and create a list of
    # set (a set is a list of each item consumed by a user, the history)
    rated = [input_batch[i][1] for i in range(len(input_batch))]
    # take the third element (the output from the dataset object) and create a stack with all the correct answers for
    # each user (the next item that will be consumed by each user), ex. dim [32]
    output = torch.stack([input_batch[i][2] for i in range(len(input_batch))], dim=0)
    return [input, rated, output]

def get_num_items(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']

def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    padding_value = stats['num_items'] + 1
    return padding_value

# configure the dataloader for the training
def get_train_dataloader(dataset_name, batch_size=32, max_length=200, train_neg_per_positive=256):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    # max lenght + 1 is useful cause in this way we can shift the items and have the last item and the corresponding
    # item to predict (ex. [A, B, C, D] the sequence +1, [A, B, C] the shifted one with the last item consumed in each
    # position, [B, C, D] the shifted one with the item to predict for each position). It's to avoid redundancy
    train_dataset = SequenceDataset(f"{dataset_dir}/train/input.txt", max_length=max_length + 1, padding_value=padding_value) # +1 for sequence shifting
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_with_random_negatives(x, padding_value , train_neg_per_positive))
    return train_loader

def get_val_or_test_dataloader(dataset_name, part='val', batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    dataset = SequenceDataset(f"{dataset_dir}/{part}/input.txt", padding_value,  f"{dataset_dir}/{part}/output.txt", max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return dataloader

def get_val_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'val', batch_size, max_length)

def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'test', batch_size, max_length)