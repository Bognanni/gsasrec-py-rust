use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use ort::ep::{self, ExecutionProvider};   // ort 2.x: ep module + trait
use std::sync::Mutex;
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
// if you need measure pure inference
// use std::time::Instant;
// use std::fs::OpenOptions;
// use std::io::Write;

use crate::model::GSASRec;
use crate::config::GsasrecConfig;


// Recommender class that uses ONNX
#[pyclass]
pub struct Recommender {
    session: Mutex<Session>,
}

#[pymethods]
impl Recommender {
    #[new]
    pub fn new(model_path: &str, device_type: &str) -> PyResult<Self> {
        let mut builder = Session::builder()
            .map_err(|e| PyRuntimeError::new_err(format!("Builder error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PyRuntimeError::new_err(format!("Optimization error: {}", e)))?
            .with_intra_threads(1)
            .map_err(|e| PyRuntimeError::new_err(format!("Intra-threads error: {}", e)))?
            .with_inter_threads(1)
            .map_err(|e| PyRuntimeError::new_err(format!("Inter-threads error: {}", e)))?;

        // ort 2.x: use ep::CUDA and the ExecutionProvider trait.
        // is_available() checks whether the ORT binary was compiled with CUDA support.
        // register() actually registers the EP on this builder and returns an error
        // if the GPU is not found at runtime — instead of silently falling back to CPU.
        if device_type == "cuda" {
            let cuda = ep::CUDA::default();

            if !cuda.is_available()
                .map_err(|e| PyRuntimeError::new_err(format!("CUDA availability check error: {}", e)))?
            {
                return Err(PyRuntimeError::new_err(
                    "CUDA execution provider is not available in this ORT build. \
                    Make sure you have the 'cuda' feature enabled in Cargo.toml \
                    and that onnxruntime-gpu is installed."
                ));
            }

            cuda.register(&mut builder)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to register CUDA EP: {}", e)))?;

            println!("ONNX Runtime initialized with CUDA.");
        } else if device_type == "cpu" {
            println!("ONNX Runtime initialized with CPU.");
        } else {
            return Err(PyRuntimeError::new_err(format!("Invalid device: '{}'. Please use 'cpu' or 'cuda'.", device_type)));
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Recommender { session: Mutex::new(session) })
    }

    pub fn get_embeddings(&self, py: Python<'_>, padded_batch: Vec<Vec<i64>>) -> PyResult<Vec<f32>> {
        // release the GIL
        py.allow_threads(move || {
            let batch_size = padded_batch.len();
            if batch_size == 0 {
                return Ok(Vec::new());
            }
            let max_length = padded_batch[0].len();

            let mut flattened_batch = Vec::with_capacity(batch_size * max_length);
            for sequence in padded_batch {
                flattened_batch.extend(sequence);
            }

            let input_shape = vec![batch_size, max_length];
            let input_tensor = Value::from_array((input_shape, flattened_batch))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let inputs = ort::inputs!["input_seq" => input_tensor];

            let mut session_lock = self.session.lock().unwrap();

            // let start_time = Instant::now();

            let outputs = session_lock.run(inputs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // let duration = start_time.elapsed();
            // let latency_ms = duration.as_secs_f64() * 1000.0;

            // if let Ok(mut file) = OpenOptions::new()
            //     .create(true)
            //     .append(true)
            //     .open("latencies.csv")
            // {
            //     let _ = writeln!(file, "{}", latency_ms);
            // }

            let (shape, data) = outputs["embedded"]
            .try_extract_tensor::<f32>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // identify the embedding dimension from the last dimension of the output shape
            let embedding_dim = *shape.last().unwrap() as usize;

            // vector to store the last token embeddings: [batch_size * embedding_dim]
            let mut last_token_embeddings = Vec::with_capacity(batch_size * embedding_dim);

            for b in 0..batch_size {
                // starting index for the last token embedding of the b-th sequence
                let start_idx = (b * max_length + (max_length - 1)) * embedding_dim;
                let end_idx = start_idx + embedding_dim;

                last_token_embeddings.extend_from_slice(&data[start_idx..end_idx]);
            }

            Ok(last_token_embeddings)
        })
    }
}


// Recommender class that uses the Rust (Candle) implementation of the model
#[pyclass]
pub struct CandleRecommender {
    model: GSASRec,
    device: Device,
}

#[pymethods]
impl CandleRecommender {
    #[new]
    pub fn new(model_path: &str, device_type: &str) -> PyResult<Self> {
        println!("Candle Runtime initialized.");

        let device = match device_type {
            "cuda" => Device::cuda_if_available(0)
                .map_err(|e| PyRuntimeError::new_err(format!("Device error: {}", e)))?,
            "cpu" => Device::Cpu,
            _ => return Err(PyRuntimeError::new_err(format!("Invalid device: '{}'. Please use 'cpu' or 'cuda'.", device_type))),
        };

        let dataset_name = "ml-1m";
        let num_items = 3416;
        let config = GsasrecConfig::new(dataset_name, num_items);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                .map_err(|e| PyRuntimeError::new_err(format!("Error loading weights: {}", e)))?
        };

        let model = GSASRec::new(vb, config)
            .map_err(|e| PyRuntimeError::new_err(format!("Error initializing model: {}", e)))?;

        Ok(CandleRecommender { model, device })
    }

    pub fn get_embeddings(&self, py: Python<'_>, padded_batch: Vec<Vec<i64>>) -> PyResult<Vec<f32>> {
        py.allow_threads(move || {
            let batch_size = padded_batch.len();
            if batch_size == 0 {
                return Ok(Vec::new());
            }
            let seq_len = padded_batch[0].len();

            let mut flattened_batch = Vec::with_capacity(batch_size * seq_len);
            for sequence in padded_batch {
                for &item in &sequence {
                    flattened_batch.push(item as u32);
                }
            }

            let input_tensor = Tensor::from_vec(flattened_batch, (batch_size, seq_len), &self.device)
                .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?;

            // let start_time = std::time::Instant::now();

            let (seq_emb, _attentions) = self.model.forward(&input_tensor, false)
                .map_err(|e| PyRuntimeError::new_err(format!("Forward pass error: {}", e)))?;

            // self.device.synchronize().map_err(|e| PyRuntimeError::new_err(format!("CUDA sync error: {}", e)))?;

            // let duration = start_time.elapsed();
            // let latency_ms = duration.as_secs_f64() * 1000.0;

            // if let Ok(mut file) = OpenOptions::new()
            //     .create(true)
            //     .append(true)
            //     .open("latencies.csv")
            // {
            //     let _ = writeln!(file, "{}", latency_ms);
            // }

            // select only the last token embedding for each sequence in the batch
            let last_token_emb = seq_emb
                .i((.., seq_len - 1, ..))
                .map_err(|e| PyRuntimeError::new_err(format!("Slicing error: {}", e)))?;

            // flatten only the last token embeddings and convert to Vec<f32>
            let output_vec = last_token_emb
                .to_device(&Device::Cpu)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .flatten_all()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(output_vec)
        })
    }
}