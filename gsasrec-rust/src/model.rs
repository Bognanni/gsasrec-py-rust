use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{embedding, layer_norm, Dropout, Embedding, LayerNorm, Module, VarBuilder};
use std::collections::HashSet;
use crate::config::GsasrecConfig;
use crate::transformer::TransformerBlock;

// GSASRec model
pub struct GSASRec {
    pub config: GsasrecConfig,
    item_embedding: Embedding,
    position_embedding: Embedding,
    embeddings_dropout: Dropout,
    transformer_blocks: Vec<TransformerBlock>,
    seq_norm: LayerNorm,
    output_embedding: Option<Embedding>,
}

impl GSASRec {
    // specify the var builder, so the weights
    pub fn new(vb: VarBuilder, config: GsasrecConfig) -> Result<Self> {
        // id corresponding to padding
        let pad_id = (config.num_items + 1) as usize;
        
        // the size is +1 again because the id=0 is unused
        let item_embedding = embedding(pad_id + 1, config.embedding_dim, vb.pp("item_embedding"))?;
        // for the position the size in input is equals to the dim. of the sequence
        let position_embedding = embedding(config.sequence_length, config.embedding_dim, vb.pp("position_embedding"))?;
        
        let embeddings_dropout = Dropout::new(config.dropout_rate);

        // a vector of transformer blocks
        let mut transformer_blocks = Vec::new();
        for i in 0..config.num_blocks {
            let block_vb = vb.pp(format!("transformer_blocks.{}", i));
            transformer_blocks.push(TransformerBlock::new(block_vb, &config)?);
        }

        let seq_norm = layer_norm(config.embedding_dim, 1e-5, vb.pp("seq_norm"))?;

        // matrix of embedding for the output if we don't reuse the item embeddings
        let output_embedding = if !config.reuse_item_embeddings {
            Some(embedding(pad_id + 1, config.embedding_dim, vb.pp("output_embedding"))?)
        } else {
            None
        };

        Ok(Self {
            config,
            item_embedding,
            position_embedding,
            embeddings_dropout,
            transformer_blocks,
            seq_norm,
            output_embedding,
        })
    }

    // return the weights of embedding depending from the reuse
    pub fn get_output_embeddings_weight(&self) -> Tensor {
        if let Some(ref out_emb) = self.output_embedding {
            out_emb.embeddings().clone()
        } else {
            self.item_embedding.embeddings().clone()
        }
    }

    pub fn forward(&self, input: &Tensor, train: bool) -> Result<(Tensor, Vec<Tensor>)> {
        // number of users
        let b_size = input.dim(0)?;
        let seq_len = input.dim(1)?;

        // get the embeddings of the input
        let mut seq = self.item_embedding.forward(input)?;

        // get the integer value associated to the padding
        let pad_val = (self.config.num_items + 1) as u32;
        // create the tensor with the value inside
        let pad_tensor = Tensor::new(pad_val, input.device())?.broadcast_as(input.shape())?;
        // obtain a mask with 0 where there is padding and 1 where there is not, unsqueeze add
        // a dimension for the next multiplications
        let mask = input.ne(&pad_tensor)?.to_dtype(DType::F32)?.unsqueeze(2)?; // Shape: [batch_size, seq_len, 1]

        // list of number from 0 to seq_len
        let positions = Tensor::arange(0u32, seq_len as u32, input.device())?
            .unsqueeze(0)?
            .broadcast_as((b_size, seq_len))?; // to have the desired dimension

        // get the embeddings
        let pos_embeddings = self.position_embedding.forward(&positions)?;
        
        // sum to consider the sequence and the position
        seq = (seq + pos_embeddings)?;
        
        // dropout if we are training
        seq = self.embeddings_dropout.forward(&seq, train)?;
        
        // multiply for the mask to not consider the padding
        seq = seq.broadcast_mul(&mask)?;

        let mut attentions = Vec::new();
        // for each block we obtain the new sequence as output and give it to the next block as input
        for block in &self.transformer_blocks {
            let (new_seq, attention) = block.forward(&seq, &mask, train)?;
            seq = new_seq;
            // save the attention matrix every time
            attentions.push(attention);
        }

        // normalize and return the embeddings of the sequences and the attentions applied
        let seq_emb = self.seq_norm.forward(&seq)?;

        Ok((seq_emb, attentions))
    }

    // function to obtain the results, not used during the training
    pub fn get_predictions(&self, input: &Tensor, limit: usize, rated: Option<&Vec<HashSet<u32>>>
    ) -> Result<Vec<Vec<(u32, f32)>>> {

        let (model_out, _) = self.forward(input, false)?;
        // number of items in the history of each user
        let seq_len = model_out.dim(1)?;

        // get the last item because thanks to the attention we have there all the history summarized
        //.contiguous() because matmul supports only contiguous tensors
        let last_item_emb = model_out.i((.., seq_len - 1, ..))?.contiguous()?;

        // take the matrix with all the items in the catalogue
        let output_weights = self.get_output_embeddings_weight().t()?.contiguous()?;
        
        // multiply with the last item to obtain a score, higher is that higher is the probability of using the item
        // for that user
        let scores_tensor = last_item_emb.matmul(&output_weights)?;
        
        // list of list to use the cpu to do the filtering
        let scores_vec: Vec<Vec<f32>> = scores_tensor.to_vec2()?;
        let pad_id = self.config.num_items + 1;

        let mut final_results = Vec::new();

        // for each user we take all the scores
        for (i, mut user_scores) in scores_vec.into_iter().enumerate() {
            // decide to not use the object 0
            user_scores[0] = f32::NEG_INFINITY;
            // if the id is equals or greater than the id of the padding we consider score - infinity too
            for idx in pad_id..user_scores.len() as u32 {
                user_scores[idx as usize] = f32::NEG_INFINITY;
            }

            // - infinity for the already consumed items too
            if let Some(rated_list) = rated {
                for &seen_item in &rated_list[i] {
                    user_scores[seen_item as usize] = f32::NEG_INFINITY;
                }
            }

            // create a vec of vec to associate each id item with the score
            let mut score_with_indices: Vec<(u32, f32)> = user_scores
                .into_iter()
                .enumerate()
                .map(|(id, score)| (id as u32, score))
                .collect();

            // sort the results using the second element of each couple (the score)
            score_with_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            score_with_indices.truncate(limit);
            final_results.push(score_with_indices);
        }

        Ok(final_results)
    }
}