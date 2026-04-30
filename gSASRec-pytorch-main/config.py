from ir_measures import nDCG, R

class GSASRecExperimentConfig(object):
    def __init__(self, dataset_name, sequence_length=200, embedding_dim=256, train_batch_size=128,
                             num_heads=4, num_blocks=3, 
                             dropout_rate=0.0,
                             negs_per_pos=256,
                             max_epochs=100,
                             max_batches_per_epoch=100,
                             metrics=[nDCG@10, R@1, R@10],
                             val_metric = nDCG@10,
                             early_stopping_patience=20,
                             gbce_t = 0.75,
                             filter_rated=True,
                             eval_batch_size=512,
                             recommendation_limit=10,
                             reuse_item_embeddings=False
                             ):
        self.sequence_length = sequence_length                  # number of old events analyzed
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads                              # number of attention heads
        self.num_blocks = num_blocks                            # number of sequential layers (how many TransformerBlock)
        self.dropout_rate = dropout_rate                        # % of neurons turned off
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size                # sequences showed to the model in a single batch of an epoch
        self.negs_per_pos = negs_per_pos                        # negative items for a single positive item
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch      # ex. train_batch_size x max_batches_per_epoch
                                                                # = sequences used for training in a single epoch
        self.val_metric = val_metric
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience  # max epochs without progression in evaluation
        self.gbce_t = gbce_t                                    # generalized binary cross entropy (specific gsasrec)
        self.filter_rated = filter_rated                        # mask items already used by the user
        self.recommendation_limit = recommendation_limit        # top k recommendation
        self.eval_batch_size = eval_batch_size
        self.reuse_item_embeddings = reuse_item_embeddings      # reuse matrix used to create embeddings in input to
                                                                # make prediction in output
        
