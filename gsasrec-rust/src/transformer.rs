use candle_core::{DType, Result, Tensor};
use candle_nn::{layer_norm, linear, Dropout, LayerNorm, Linear, Module, VarBuilder};
use crate::config::GsasrecConfig;

// Multi head attention sasrec style
pub struct MultiHeadAttention {
    num_heads: usize,
    query_proj: Linear,
    key_proj: Linear,
    val_proj: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    // constructor of the multi head attention
    pub fn new(vb: VarBuilder, config: &GsasrecConfig) -> Result<Self> {
        let dim = config.embedding_dim;
        let num_heads = config.num_heads;
        
        // initialize the linear network to obtain query, key and value
        let query_proj = linear(dim, dim, vb.pp("query_proj"))?;
        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        // dropout to turn off the neurons
        let dropout = Dropout::new(config.dropout_rate);

        Ok(Self { num_heads, query_proj, key_proj, val_proj, dropout })
    }

    // forward to obtain the output of the multi head attention and the weights of the attentions
    pub fn forward(&self, queries: &Tensor, keys: &Tensor, causality: bool, train: bool) -> Result<(Tensor, Tensor)> {
        // obtain the tensors, output of the linear networks
        let q = self.query_proj.forward(queries)?;
        let k = self.key_proj.forward(keys)?;
        let v = self.val_proj.forward(keys)?;

        // chunks the dimension of the embeddings returning a list of Tensor long as the num_heads, then concatenate
        // this for q, k and v
        // shapes -> q = [batch_size, sequence_length, embedding_dim], q_chunks = list of [batch_size, sequence_length, head_dim], length = num_heads
        // q_ = [batch_size * num_heads, sequence_length, head_dim]
        let q_chunks = q.chunk(self.num_heads, 2)?;
        let q_ = Tensor::cat(&q_chunks, 0)?;
        
        let k_chunks = k.chunk(self.num_heads, 2)?;
        let k_ = Tensor::cat(&k_chunks, 0)?;
        
        let v_chunks = v.chunk(self.num_heads, 2)?;
        let v_ = Tensor::cat(&v_chunks, 0)?;

        // prepare k for the mul with q and follow the self attention formula, with scale
        let k_t = k_.transpose(1, 2)?.contiguous()?;
        let mut outputs = q_.matmul(&k_t)?;

        let scale = (k_.dim(2)? as f64).sqrt();
        outputs = (outputs / scale)?;

        // create a mask to find the items that are padding (0) or not (1) summing the value of the embeddings
        let key_sum = keys.abs()?.sum_keepdim(2)?;
        
        let zero_tensor = Tensor::new(0.0f32, keys.device())?.broadcast_as(key_sum.shape())?;
        
        let key_masks = key_sum.gt(&zero_tensor)?.to_dtype(DType::F32)?;
        
        // same shape of the q, k, v, with the heads concatenate
        let mut key_masks_chunks = Vec::new();
        for _ in 0..self.num_heads { key_masks_chunks.push(key_masks.clone()); }
        let key_masks_repeated = Tensor::cat(&key_masks_chunks, 0)?;
        
        let seq_len = queries.dim(1)?;
        let key_masks_final = key_masks_repeated.transpose(1, 2)?
            .broadcast_as((q_.dim(0)?, seq_len, k_.dim(1)?))?; //modify the shape to be equals to the outputs

        // -infinity applied where there are padding items using the mask, elsewhere left the value computed
        let zero_tensor_k = Tensor::new(0.0f32, outputs.device())?.broadcast_as(key_masks_final.shape())?;
        let is_zero = key_masks_final.eq(&zero_tensor_k)?;
        let neg_inf_tensor = Tensor::new(f32::NEG_INFINITY, outputs.device())?.broadcast_as(outputs.shape())?;
        
        outputs = is_zero.where_cond(&neg_inf_tensor, &outputs)?;

        // if there is causality (always) we put -infinity for the items that see at the future (not fair)
        if causality {
            let mut mask_data = vec![0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i { mask_data[i * seq_len + j] = f32::NEG_INFINITY; }
                }
            }
            let causal_mask = Tensor::from_vec(mask_data, (1, seq_len, seq_len), outputs.device())?
                .broadcast_as(outputs.shape())?;
            outputs = (outputs + causal_mask)?;
        }

        // softmax to have values between 0 and 1
        outputs = candle_nn::ops::softmax_last_dim(&outputs)?;
        
        // where there is nAn (softmax return nAn if an entire row was padding) put 0
        let is_nan = outputs.ne(&outputs)?;
        let zeros = Tensor::zeros_like(&outputs)?;
        outputs = is_nan.where_cond(&zeros, &outputs)?;

        // check if there are padding items summing the abs values, keepdim is to maintain the dimension
        // ex. from [32, 200, 256] to [32, 200, 1]
        let query_sum = queries.abs()?.sum_keepdim(2)?;
        let zero_tensor_q = Tensor::new(0.0f32, queries.device())?.broadcast_as(query_sum.shape())?;
    
        // check which value is greater than 0 and substitute the values in the last dimension with 0 or 1
        let query_masks = query_sum.gt(&zero_tensor_q)?.to_dtype(DType::F32)?;

        let mut query_masks_chunks = Vec::new();
        // same operation done before, divide for each head and then concat
        for _ in 0..self.num_heads { query_masks_chunks.push(query_masks.clone()); }
        let query_masks_repeated = Tensor::cat(&query_masks_chunks, 0)?;
        
        outputs = outputs.broadcast_mul(&query_masks_repeated)?;

        // stack to see better the attentions
        let att_chunks = outputs.chunk(self.num_heads, 0)?;
        let attention_weights = Tensor::stack(&att_chunks, 1)?;

        // dropout to the outputs and multiply the attentions to the value
        outputs = self.dropout.forward(&outputs, train)?;

        outputs = outputs.matmul(&v_)?;
        
        // divide again the results and concatenate to have the original dimension of the input tensor
        let out_chunks = outputs.chunk(self.num_heads, 0)?;
        outputs = Tensor::cat(&out_chunks, 2)?;

        Ok((outputs, attention_weights))
    }
}

// struct to put all together
pub struct TransformerBlock {
    first_norm: LayerNorm,
    second_norm: LayerNorm,
    multihead_attention: MultiHeadAttention,
    dense1: Linear, // 1 feed-forward network
    dense2: Linear, // 2 feed-forward network
    dropout: Dropout,
    causality: bool,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder, config: &GsasrecConfig) -> Result<Self> {
        let dim = config.embedding_dim;
        let hidden_dim = config.embedding_dim;

        let first_norm = layer_norm(dim, 1e-5, vb.pp("first_norm"))?;
        let second_norm = layer_norm(dim, 1e-5, vb.pp("second_norm"))?;
        let multihead_attention = MultiHeadAttention::new(vb.pp("multihead_attention"), config)?;
        
        let dense1 = linear(dim, hidden_dim, vb.pp("dense1"))?;
        let dense2 = linear(hidden_dim, dim, vb.pp("dense2"))?;
        let dropout = Dropout::new(config.dropout_rate);

        Ok(Self {
            first_norm,
            second_norm,
            multihead_attention,
            dense1,
            dense2,
            dropout,
            causality: true
        })
    }

    pub fn forward(&self, seq: &Tensor, mask: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let x = self.first_norm.forward(seq)?;
        let queries = &x;
        // keys = seq not normalized following the paper
        let keys = seq;

        let (mut x, attentions) = self.multihead_attention.forward(queries, keys, self.causality, train)?;

        x = (x + queries)?;
        x = self.second_norm.forward(&x)?;

        // current x copied in residual
        let residual = x.clone();
        x = self.dense1.forward(&x)?;
        x = x.relu()?;
        x = self.dropout.forward(&x, train)?;
        x = self.dense2.forward(&x)?;
        x = self.dropout.forward(&x, train)?;

        x = (x + residual)?;

        // final output with the growing context in each item
        x = x.broadcast_mul(mask)?;

        Ok((x, attentions))
    }
}