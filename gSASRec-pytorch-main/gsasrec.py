import torch 
from transformer_decoder import TransformerBlock

class GSASRec(torch.nn.Module):
    def __init__ (self, num_items, sequence_length=200, embedding_dim=256, num_heads=4, num_blocks=3, dropout_rate=0.5, reuse_item_embeddings=False):
        super(GSASRec, self).__init__()
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)

        self.num_heads = num_heads

        # lookup table to save the embeddings (num_items size of the dictionary and embedding_dim dimension of each emb)
        self.item_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim) # items are enumerated from 1;  +1 for padding
        # lookup table for the positions (0-199 if sequence_length=200)
        self.position_embedding = torch.nn.Embedding(self.sequence_length, self.embedding_dim)

        # list with the independent copies of transformer_blocks
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)
        # decide to use the same matrix to embed the input and to make prevision in output or not
        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)

    def get_output_embeddings(self) -> torch.nn.Embedding:
        if self.reuse_item_embeddings:
            return self.item_embedding
        else:
            return self.output_embedding

    # returns last hidden state and the attention weights
    def forward(self, input):
        # convert the tensor into the 3d tensor of shape [batch_size, sequence_length, embedding_dim]
        seq = self.item_embedding(input.long())
        # mask where 1 = real item, 0 = padding
        mask = (input != self.num_items + 1).float().unsqueeze(-1)

        # fuse the input and the position
        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)[:input.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        # use the blocks as the output of the one is the input of the next one
        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            seq, attention = block(seq, mask)
            attentions.append(attention)
        
        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions
    
    def get_predictions(self, input, limit, rated=None):
        with torch.no_grad():
            # feed forward the network to obtain the output
            model_out, _ = self.forward(input)
            # take the last item
            seq_emb = model_out[:,-1,:] 
            output_embeddings = self.get_output_embeddings()
            # matrix mul between the user and the matrix of all the item [batch_size x dim] x [num_items x dim]
            # = [batch_size x dim]
            # einsum allow to describe what we want, in this case bd, nd -> bn, the dimensions we have and we want in output
            scores = torch.einsum('bd,nd->bn', seq_emb, output_embeddings.weight)
            # avoid to recommend past items or fake ones
            scores[:,0] = float("-inf")
            scores[:,self.num_items+1:] = float("-inf")
            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float("-inf")
            # return top k items
            result = torch.topk(scores, limit, dim=1)
            return result.indices, result.values
