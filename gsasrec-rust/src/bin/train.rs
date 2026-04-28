use candle_core::{DType, Result, Tensor, Device};
use candle_nn::{AdamW, Embedding, Optimizer, ParamsAdamW, VarBuilder, VarMap, Module};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use std::time::Instant;


use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::dataset::{get_padding_value, get_train_batch, SequenceDataset};
use gsasrec_rust::model::GSASRec;
use gsasrec_rust::eval::evaluate;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let models_dir = "models";
    // if the dir does not exist
    if fs::metadata(models_dir).is_err() {
        // create dir if it is possible
        fs::create_dir(models_dir).expect("Impossible to create the directory.");
    }

    let dataset_name = "ml-1m";
    let num_items = 3416;
    let config = GsasrecConfig::new(dataset_name, num_items);
    let pad_val = get_padding_value("datasets/ml1m");

    let train_dataset = SequenceDataset::new("datasets/ml1m/train/input.txt",pad_val,
        None, config.sequence_length + 1);

    let val_dataset = SequenceDataset::new("datasets/ml1m/val/input.txt", pad_val,
        Some("datasets/ml1m/val/output.txt"), config.sequence_length);

    let mut train_indices: Vec<usize> = (0..train_dataset.inputs.len()).collect();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GSASRec::new(vb, config.clone())?;

    let mut optimizer = AdamW::new(varmap.all_vars(),ParamsAdamW {lr: 0.001, ..Default::default()})?;

    let batches_per_epoch = std::cmp::min(config.max_batches_per_epoch, train_indices.len() / config.train_batch_size);
    
    // 'best' variables initialized
    let mut best_metric = f32::NEG_INFINITY;
    let mut best_model_name = String::new();
    let mut step = 0;
    let mut steps_not_improved = 0;

    println!("Starting training (Max number of epochs: {})", config.max_epochs);

    for epoch in 0..config.max_epochs {
        let start_time = Instant::now();
        let mut loss_sum = 0.0;
        
        train_indices.shuffle(&mut thread_rng());

        for batch_idx in 0..batches_per_epoch {
            step += 1;
            
            let start_idx = batch_idx * config.train_batch_size;
            let end_idx = start_idx + config.train_batch_size;
            let batch_indices = &train_indices[start_idx..end_idx];

            // get the training batch
            let batch = get_train_batch(&train_dataset, batch_indices, config.negs_per_pos, &device)?;
            let positives = batch.inputs;
            let negatives = batch.negatives;

            // to apply data shifting
            let seq_len = positives.dim(1)? - 1;

            // take all the items but not the last one that will be the target
            let model_input = positives.narrow(1, 0, seq_len)?;
            
            // take all the items but not the first one, in this way for the same index i we have the last item for
            // training in model_input and the target in labels
            let labels = positives.narrow(1, 1, seq_len)?;
            // we start from 1 because for each position we have the negatives for the item associated in model_input seen
            // as a target, and the first is not affected
            let negatives = negatives.narrow(1, 1, seq_len)?;

            // take only the embeddings, not the attentions
            let (last_hidden_state, _) = model.forward(&model_input, true)?;

            // unsqueeze to have the same dim and concat pos and neg
            let labels_unsqueezed = labels.unsqueeze(2)?;
            let pos_neg_concat = Tensor::cat(&[&labels_unsqueezed, &negatives], 2)?;

            // get the weights to compute the embeddings of the list of positives and negatives
            let out_weights = model.get_output_embeddings_weight();

            // create the Embeddings object and apply it to pos_neg_concat
            let temp_out_emb = Embedding::new(out_weights, config.embedding_dim);
            let pos_neg_embeddings = temp_out_emb.forward(&pos_neg_concat)?;

            // tensor with only the pad tensor value (num_items + 1), shape: [batch_size, sequence_length]
            let pad_tensor = Tensor::new((num_items + 1) as u32, &device)?.broadcast_as(model_input.shape())?;
            // tensor mask comparing the value of the input and the value for the padding (1 if different, 0 if equals)
            let mask = model_input.ne(&pad_tensor)?.to_dtype(DType::F32)?;

            let b_size = model_input.dim(0)?;
            // reshape the embeddings after the forward in the model to have this shape (fuse the batch size and the sequence
            // length into a dimension)
            let lhs_reshaped = last_hidden_state.reshape((b_size * seq_len, 1, config.embedding_dim))?;
            
            // reshape the embeddings of positives and negatives and transpose the 1 and 2 dims (1 + config.negs_per_pos, config.embedding_dim)
            let pne_reshaped = pos_neg_embeddings.reshape((b_size * seq_len, 1 + config.negs_per_pos, config.embedding_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            
            // dot product to compute the affinity between each user and each item
            // [batch*seq, 1, dim] x [batch*seq, dim, 1+negs] -> [batch*seq, 1, 1+negs]
            // then reshape and in the last dimension we will have the first item positive and the others negatives, for each
            // one the raw score computed before indicating the affinity
            let logits = lhs_reshaped.matmul(&pne_reshaped)?
                .reshape((b_size, seq_len, 1 + config.negs_per_pos))?;

            // gBCE (Generalized Binary Cross Entropy)
            // ground_truth: [batch, seq, 1+negs] with 1 where there is the first item (positive) 
            // and 0 the others (negatives)
            // (actually it is a single vec with dim batch * seq * (1 + negs))
            let mut gt_data = vec![0.0f32; b_size * seq_len * (1 + config.negs_per_pos)];
            for i in 0..(b_size * seq_len) {
                gt_data[i * (1 + config.negs_per_pos)] = 1.0;
            }
            // cast into Tensor with the same shape of logits
            let gt = Tensor::from_vec(gt_data, logits.shape(), &device)?;

            // scores of the positive and negative logits
            let positive_logits = logits.narrow(2, 0, 1)?;
            let negative_logits = logits.narrow(2, 1, config.negs_per_pos)?;

            // probability of a random item considered negative
            let alpha = config.negs_per_pos as f64 / (num_items as f64 - 1.0);
            // temperature hyperparameter
            let t = config.gbce_t as f64;
            // exponent used computed using alpha and t
            let beta = alpha * ((1.0 - 1.0 / alpha) * t + 1.0 / alpha);
            let eps = 1e-7;

            // sigmoid to have the scores between 0 and 1
            let positive_probs = candle_nn::ops::sigmoid(&positive_logits)?;
            // set the max and min to not be exactly 0 or 1
            let positive_probs = positive_probs.maximum(eps)?.minimum(1.0 - eps)?;
            
            // pow of -beta to use the penality computed and set the limits again
            let pos_probs_adjusted = positive_probs.to_dtype(DType::F64)?.powf(-beta)?.maximum(1.0 + eps)?.minimum(f64::MAX)?;

            // 1 / (pos_probs_adjusted - 1) to have again logits (after we compute the log)
            let to_log = (Tensor::new(1.0f64, &device)?.broadcast_as(pos_probs_adjusted.shape())?
                / (pos_probs_adjusted - 1.0)?)?.maximum(eps)?.minimum(f64::MAX)?;
            
            let positive_logits_transformed = to_log.log()?.to_dtype(DType::F32)?;
            
            // concat the positive logits with the mitigate bias and the negative logits
            let final_logits = Tensor::cat(&[&positive_logits_transformed, &negative_logits], 2)?;

            // BCEWithLogits: max(x, 0) - x*z + log(1 + exp(-abs(x)))
            let relu_logits = final_logits.relu()?;
            let x_times_z = final_logits.broadcast_mul(&gt)?;
            let neg_abs_logits = (final_logits.abs()? * -1.0)?;
            let log_term = (neg_abs_logits.exp()? + 1.0)?.log()?;
            
            // loss for each element and for each user
            let loss_per_element = ((relu_logits - x_times_z)? + log_term)?;
            
            // mean to have the loss per seq and dot product with the mask to have 0 where there is padding
            let loss_per_seq = loss_per_element.mean(2)?.broadcast_mul(&mask)?;
            
            // the global loss
            let mask_sum = mask.sum_all()?.to_scalar::<f32>()?;
            let loss = (loss_per_seq.sum_all()? / mask_sum as f64)?;

            // backprop and sum the loss
            optimizer.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        println!("Epoch {}/{} finished in {:.2}s. Loss media: {:.4}", epoch + 1, config.max_epochs,
        elapsed, loss_sum / batches_per_epoch as f32);

        // evaluate the results
        let eval_results = evaluate(&model, &val_dataset, config.eval_batch_size, &device,
            &config.metrics, config.recommendation_limit, config.filter_rated)?;

        println!("Evaluation results:");
        for (metric, score) in &eval_results {
            println!("{:?}: {:.4}", metric, score);
        }

        // value of the metric to choose the better model
        let current_val_metric = *eval_results.get(&config.val_metric).unwrap();

        if current_val_metric > best_metric {
            best_metric = current_val_metric;
            steps_not_improved = 0;
            
            let model_name = format!("{}/gsasrec-{}-step{}-best.safetensors", models_dir, dataset_name, step);
            if !best_model_name.is_empty() {
                let _ = fs::remove_file(&best_model_name);
            }
            best_model_name = model_name.clone();
            
            varmap.save(&model_name)?;
            println!("New better model saved.");
        } else {
            steps_not_improved += 1;
            println!("No better perfomances for {} step.", steps_not_improved);
            if steps_not_improved >= config.early_stopping_patience {
                println!("Early Stopping. Stop training.");
                break;
            }
        }
    }

    Ok(())
}
