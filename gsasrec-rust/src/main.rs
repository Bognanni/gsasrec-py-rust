use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use std::error::Error;

// function to prepare the sequence adding padding or removing oldest items
fn prepare_sequence(sequence: &[i64], max_length: usize, padding_value: i64) -> Vec<i64> {
    let mut prepared = Vec::new();
    let seq_len = sequence.len();

    if seq_len > max_length {
        // take the last items if the sequence is too long (using slicing)
        prepared.extend_from_slice(&sequence[(seq_len - max_length)..]);
    } else {
        
        let pad_len = max_length - seq_len;
        // fill the vec with a number of padding_value (usually 0) equals to pad_len
        prepared.resize(pad_len, padding_value);
        // finish to fill with the sequence
        prepared.extend_from_slice(sequence);
    }
    prepared
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = "models/gsasrec-ml1m-step_4465-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.019461369383357036.onnx";
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)? // optimized import (not mandatory)
        .commit_from_file(model_path)?;

    println!("Loaded model.");

    let max_length = 50;
    let padding_value = 0;

    // example of user history
    let user_history = vec![15, 82, 104];
    let padded_history = prepare_sequence(&user_history, max_length, padding_value);

    // shape of the input (1 because we have a batch of 1 user)
    let input_shape = vec![1, max_length];

    // cast the vec into a tensor understandable by ort, considering the shape and the input
    let input_tensor = Value::from_array((input_shape, padded_history))?;

    // limit value in input for top k
    let limit_value = 10i64;
    let limit_tensor = Value::from_array((vec![1], vec![limit_value]))?;

    // mapping the input with exactly the input name used in python during the export phase
    let inputs = ort::inputs!["input_seq" => input_tensor, "limit" => limit_tensor];

    // mapping also the output after running the model
    let outputs = session.run(inputs)?;

    // extract the output specifying the output type
    let indices = outputs["indices"].try_extract_tensor::<i64>()?;
    let values = outputs["values"].try_extract_tensor::<f32>()?;
    
    let items_slice = indices.1;
    let scores_slice = values.1;

    println!("Shapes results: {}, {}", indices.0, values.0);

    // top k results
    for i in 0..limit_value as usize {
        println!("Rank {}: Item ID = {}, Score = {:.4}", i + 1, items_slice[i], scores_slice[i]);
    }

    Ok(())
}
