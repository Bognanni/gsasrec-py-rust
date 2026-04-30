from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    train_batch_size=32, 
    sequence_length=50, 
    embedding_dim=128,
    num_heads=2,
    max_batches_per_epoch=100,
    max_epochs=2,
    num_blocks=2,
    dropout_rate=0.323509645777491,
    negs_per_pos=256,
    gbce_t=0.75,
)
