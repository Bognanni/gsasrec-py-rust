from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    train_batch_size=64, # half the default cause out of memory
    sequence_length=50, # from 200 to 50 for the same reason
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t = 0.75,
)
