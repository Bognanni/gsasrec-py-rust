use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::model::GSASRec;

fn main() {
    println!("Start pure latency test.");

    let device = Device::cuda_if_available(0).expect("Critical error: Unable to initialize CUDA/CPU device");
    let model_path = "model.safetensors";
    let num_items = 3416;
    let sequence_length = 200;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
            .expect("Critical error: Unable to read safetensors file")
    };

    let mut config = GsasrecConfig::new("infer", num_items as u32);
    config.sequence_length = sequence_length;
    let model = GSASRec::new(vb, config).expect("Critical error: Unable to create model");

    let batch_size = 40;
    
    let flat_dummy_data = vec![1u32; batch_size * sequence_length];
    
    let input_tensor = Tensor::from_vec(flat_dummy_data, (batch_size, sequence_length), &device)
        .expect("Critical error: Unable to create input tensor");

    println!("Model loaded. Input Tensor Shape: [{}, {}]", batch_size, sequence_length);

    println!("Warm-up - 30 iterations of forward pass.");
    for _ in 0..30 {
        let _ = model.forward(&input_tensor, false).unwrap();
    }

    let iterations = 1000;
    let mut times_ms = Vec::with_capacity(iterations);

    println!("Executing {} iterations of forward pass...", iterations);
    
    let total_start = Instant::now();

    for _ in 0..iterations {
        let t0 = Instant::now();
        
        let (_seq_emb, _attentions) = model.forward(&input_tensor, false).unwrap();

        device.synchronize().expect("Error during CUDA synchronization");
        times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = times_ms.iter().sum();
    let avg = sum / iterations as f64;
    let fastest = times_ms[0];
    let slowest = times_ms[iterations - 1];

    let p = |percentile: f64| -> f64 {
        let idx = ((iterations as f64 * percentile) / 100.0).round() as usize;
        let idx = idx.clamp(0, iterations - 1);
        times_ms[idx]
    };

    println!("\nBenchmark Results:");
    println!("Total Execution Time:  {:.2} seconds", total_elapsed);
    println!("Iterations:               {}", iterations);
    println!("Average:                  {:.3} ms", avg);
    println!("Fastest (Min):            {:.3} ms", fastest);
    println!("Slowest (Max):            {:.3} ms", slowest);
    println!("---------------------------------------");
    println!("P50:                      {:.3} ms", p(50.0));
    println!("P90:                      {:.3} ms", p(90.0));
    println!("P95:                      {:.3} ms", p(95.0));
    println!("P99:                      {:.3} ms", p(99.0));
    println!("P99.9:                    {:.3} ms", p(99.9));
}