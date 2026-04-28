use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use std::env;

use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::dataset::{get_padding_value, SequenceDataset};
use gsasrec_rust::eval::evaluate;
use gsasrec_rust::model::GSASRec;

// use "cargo run --bin test -- <path_model>"
fn main() -> Result<()> {
    println!("Test the model");

    // args from the command line
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Error: must write the path of the model");
        println!("Use: cargo run --bin test -- <path_model>");
        return Ok(());
    }
    let checkpoint_path = &args[1];

    let device = Device::new_cuda(0)?;
    let dataset_name = "ml1m";
    let num_items = 3416;
    
    let config = GsasrecConfig::new(dataset_name, num_items);
    let pad_val = get_padding_value(&format!("datasets/{}", dataset_name));

    // empty model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GSASRec::new(vb, config.clone())?;

    // load the weights saved
    println!("Loading weights from: {}", checkpoint_path);
    varmap.load(checkpoint_path)?;

    // prepare the dataloader
    let test_dataset = SequenceDataset::new(&format!("datasets/{}/test/input.txt", dataset_name),
        pad_val, Some(&format!("datasets/{}/test/output.txt", dataset_name)),
        config.sequence_length);

    // Evaluation results
    let evaluation_result = evaluate(&model, &test_dataset, config.eval_batch_size, &device,
        &config.metrics, config.recommendation_limit, config.filter_rated)?;

    // print the results
    println!("\nResults on Test Set:");
    for (metric, score) in &evaluation_result {
        println!("{:?}: {:.4}", metric, score);
    }

    Ok(())
}