from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    train_batch_size=64,
    sequence_length=200,
    embedding_dim=256,
    num_heads=4,
    max_batches_per_epoch=100,
    max_epochs=100,
    num_blocks=1,
    dropout_rate=0.16519583830077267,
    negs_per_pos=256,
    gbce_t=0.5,
)
