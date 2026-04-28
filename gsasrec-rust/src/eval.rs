use candle_core::{Device, Result};
use std::collections::HashMap;

use crate::config::Metric;
use crate::dataset::{get_eval_batch, SequenceDataset};
use crate::model::GSASRec;

// function to compute the metrics on the val o test set
pub fn evaluate(model: &GSASRec, dataset: &SequenceDataset, batch_size: usize, device: &Device, metrics: &[Metric],
    limit: usize, filter_rated: bool) -> Result<HashMap<Metric, f32>> {
    
    // Hashmap to save the value for each metric, initialize with 0
    let mut metric_sums: HashMap<Metric, f32> = HashMap::new();
    for metric in metrics {
        metric_sums.insert(metric.clone(), 0.0);
    }
    
    let total_users = dataset.inputs.len();
    let num_batches = (total_users + batch_size - 1) / batch_size;

    // for each batch
    for b in 0..num_batches {
        let start_idx = b * batch_size;
        // useful to don't go over the number of users if it is the last batch
        let end_idx = std::cmp::min(start_idx + batch_size, total_users);
        let indices: Vec<usize> = (start_idx..end_idx).collect();

        // get the eval batch with history and future targets
        let batch = get_eval_batch(dataset, &indices, device)?;
        
        // rated items (single items consumed by each user) if needed
        let rated_opt = if filter_rated { Some(&batch.rated) } else { None };
        let predictions = model.get_predictions(&batch.inputs, limit, rated_opt)?;
        
        // outputs (the target) into a vec
        let targets = batch.outputs.to_vec1::<u32>()?;

        // iterate for each couple (prediction, target) associated with each user in the batch
        for (user_preds, target) in predictions.iter().zip(targets.iter()) {
            
            // get the ranking of the last item actually consumed
            let mut rank: Option<usize> = None;
            for (r, (item_id, _score)) in user_preds.iter().enumerate() {
                if item_id == target {
                    // r is the ranking predicted
                    rank = Some(r + 1); // +1 to start from 1
                    break;
                }
            }

            // update each metric
            for metric in metrics {
                match metric {
                    Metric::NDCG(k) => {
                        if let Some(r) = rank {
                            // check if the ranking predicted is lower than the limit k (* to dereference)
                            if r <= *k {
                                // Formula NDCG: 1 / log2(rank + 1)
                                let score = 1.0 / ((r as f32 + 1.0).log2());
                                // * because get_mut returns a pointer so we have to dereference
                                *metric_sums.get_mut(metric).unwrap() += score;
                            }
                        }
                    }
                    Metric::Recall(k) => {
                        if let Some(r) = rank {
                            if r <= *k {
                                // Hit Ratio / Recall: 1 if is in the top k
                                *metric_sums.get_mut(metric).unwrap() += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // mean considering all the users, for NDCG and for Recall
    let mut final_metrics = HashMap::new();
    for (metric, sum) in metric_sums {
        final_metrics.insert(metric, sum / total_users as f32);
    }

    Ok(final_metrics)
}