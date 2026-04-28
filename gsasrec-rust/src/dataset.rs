use candle_core::{Device, Result, Tensor};
use rand::Rng;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};


// macro to generate implementations of the Deserialize traits for data structure
#[derive(Deserialize)]
pub struct DatasetStats {
    pub num_items: u32,
}

// function to ge the dataset stats from the json file
pub fn get_dataset_stats(dataset_dir: &str) -> DatasetStats {
    let file = File::open(format!("{}/dataset_stats.json", dataset_dir))
        .expect("File dataset_stats.json not found.");
    let reader = BufReader::new(file);
    
    // returns the data in a DatasetStats struct
    serde_json::from_reader(reader).expect("Error during the reading.")
}

// get the value associated to padding (number of items + 1)
pub fn get_padding_value(dataset_dir: &str) -> u32 {
    let stats = get_dataset_stats(dataset_dir);
    stats.num_items + 1
}

// dataset struct
pub struct SequenceDataset {
    pub inputs: Vec<Vec<u32>>,
    pub outputs: Option<Vec<u32>>,  // Option because there are no outputs for training
    pub max_length: usize,
    pub padding_value: u32,
}

impl SequenceDataset {
    pub fn new(input_file: &str, padding_value: u32, output_file: Option<&str>, max_length: usize) -> Self {
        let file = File::open(input_file).expect("Error. input_file not opened.");
        // list of lists to take all the inputs
        let inputs: Vec<Vec<u32>> = BufReader::new(file)
            .lines()
            .map(|line| {
                line.unwrap()
                    .split_whitespace()
                    .filter_map(|s| s.parse::<u32>().ok())
                    .collect()
            })
            .collect();

        // a number for each user (the last item)
        let outputs = output_file.map(|path| {
            let file = File::open(path).expect("Error. output_file not opened.");
            BufReader::new(file)
                .lines()
                .map(|line| line.unwrap().trim().parse::<u32>().unwrap())
                .collect()
        });

        Self { inputs, outputs, max_length, padding_value }
    }

    // returns the padded history of a user and all the items consumed without repetition (the HashSet)
    pub fn get_item(&self, idx: usize) -> (Vec<u32>, HashSet<u32>) {
        // copy of the history
        let mut inp = self.inputs[idx].clone();

        // copy of the history without repetition
        let rated: HashSet<u32> = inp.iter().cloned().collect();

        // more recent items if the history is longer than the max_length
        if inp.len() > self.max_length {
            inp = inp[(inp.len() - self.max_length)..].to_vec();
        }
        // add padding at the beginning if is shorter
        else if inp.len() < self.max_length {
            let diff = self.max_length - inp.len();
            let mut padded = vec![self.padding_value; diff];
            padded.extend(inp);
            inp = padded;
        }

        (inp, rated)
    }
}

// batches for training and evaluation/test
pub struct TrainBatch {
    pub inputs: Tensor,
    pub negatives: Tensor,
}

pub struct EvalBatch {
    pub inputs: Tensor,
    pub rated: Vec<HashSet<u32>>,
    pub outputs: Tensor,
}

// populates and returns the train batch
pub fn get_train_batch(dataset: &SequenceDataset, indices: &[usize], num_negatives: usize, device: &Device)
    -> Result<TrainBatch> {
    let mut batch_inputs = Vec::new();
    let mut rng = rand::thread_rng();
    
    let batch_size = indices.len();
    let seq_len = dataset.max_length;

    // all the sequences for all users in a single list
    for &idx in indices {
        let (inp, _) = dataset.get_item(idx);
        batch_inputs.extend(inp);
    }

    // num_negatives for each positive item for each user
    let total_negatives = batch_size * seq_len * num_negatives;
    let mut negatives_vec = Vec::with_capacity(total_negatives);
    for _ in 0..total_negatives {
        let neg_id = rng.gen_range(1..dataset.padding_value);
        // all in one list
        negatives_vec.push(neg_id);
    }

    // Vec casted into tensor with batch_size rows and seq_len columns
    let inputs_tensor = Tensor::from_vec(batch_inputs, (batch_size, seq_len), device)?;
    
    // for the negative tensor we have +1 dimension
    let negatives_tensor = Tensor::from_vec(negatives_vec, (batch_size, seq_len, num_negatives), device)?;

    Ok(TrainBatch { inputs: inputs_tensor, negatives: negatives_tensor })
}

// populates and returns the eval/test batch
pub fn get_eval_batch(dataset: &SequenceDataset, indices: &[usize], device: &Device) -> Result<EvalBatch> {
    let mut batch_inputs = Vec::new();
    let mut batch_rated = Vec::new();
    // batch output to test the model
    let mut batch_outputs = Vec::new();
    
    let batch_size = indices.len();
    let seq_len = dataset.max_length;

    for &idx in indices {
        let (inp, rated) = dataset.get_item(idx);
        batch_inputs.extend(inp);
        batch_rated.push(rated);
        
        if let Some(ref outs) = dataset.outputs {
            batch_outputs.push(outs[idx]);
        }
    }

    let inputs_tensor = Tensor::from_vec(batch_inputs, (batch_size, seq_len), device)?;
    let outputs_tensor = Tensor::from_vec(batch_outputs, batch_size, device)?;

    // returns also the batch_rated (single items consumed by each user)
    Ok(EvalBatch { inputs: inputs_tensor, rated: batch_rated, outputs: outputs_tensor })
}