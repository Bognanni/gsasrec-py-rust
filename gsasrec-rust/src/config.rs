// enum for metrics used
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    NDCG(usize),
    Recall(usize),
}

// struct to save the config of the model
#[derive(Clone, Debug)]
pub struct GsasrecConfig {
    pub dataset_name: String,
    pub num_items: u32, // added to create the embedding
    
    // model parameters
    pub sequence_length: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub dropout_rate: f32,
    pub reuse_item_embeddings: bool,
    
    // training parameters
    pub train_batch_size: usize,
    pub negs_per_pos: usize,
    pub max_epochs: usize,
    pub max_batches_per_epoch: usize,
    pub gbce_t: f32, // Generalized Binary Cross Entropy t
    
    // evaluation parameters
    pub metrics: Vec<Metric>,
    pub val_metric: Metric,
    pub early_stopping_patience: usize,
    pub filter_rated: bool,
    pub eval_batch_size: usize,
    pub recommendation_limit: usize,
}

impl GsasrecConfig {
    pub fn new(dataset_name: &str, num_items: u32) -> Self {
        Self {
            dataset_name: dataset_name.to_string(),
            num_items,
            
            sequence_length: 50,
            embedding_dim: 256,
            num_heads: 4,
            num_blocks: 3,
            dropout_rate: 0.0,
            reuse_item_embeddings: false,
            
            train_batch_size: 16,
            negs_per_pos: 256,
            max_epochs: 50,
            max_batches_per_epoch: 100,
            gbce_t: 0.75,
            
            metrics: vec![Metric::NDCG(10), Metric::Recall(1), Metric::Recall(10)],
            val_metric: Metric::NDCG(10),
            
            early_stopping_patience: 200,
            filter_rated: true,
            eval_batch_size: 512,
            recommendation_limit: 10,
        }
    }
}