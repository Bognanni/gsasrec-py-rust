use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use ort::ep::{self, ExecutionProvider};   // ort 2.x: ep module + trait
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;

use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;

use crate::model::GSASRec;
use crate::config::GsasrecConfig;


// Recommender class that uses ONNX
#[pyclass]
pub struct Recommender {
    session: Session,
}

#[pymethods]
impl Recommender {
    #[new]
    pub fn new(model_path: &str) -> PyResult<Self> {
        let mut builder = Session::builder()
            .map_err(|e| PyRuntimeError::new_err(format!("Builder error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PyRuntimeError::new_err(format!("Optimization error: {}", e)))?;

        // ort 2.x: use ep::CUDA and the ExecutionProvider trait.
        // is_available() checks whether the ORT binary was compiled with CUDA support.
        // register() actually registers the EP on this builder and returns an error
        // if the GPU is not found at runtime — instead of silently falling back to CPU.
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

        // register() propagates the error if the GPU cannot be initialised,
        // so we get an explicit failure instead of a silent CPU fallback.
        cuda.register(&mut builder)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to register CUDA EP: {}", e)))?;

        println!("ONNX Runtime initialized with CUDA.");

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Recommender { session })
    }

    pub fn get_embeddings(&mut self, padded_batch: Vec<Vec<i64>>) -> PyResult<Vec<f32>> {
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

        let start_time = Instant::now();

        let outputs = self.session.run(inputs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let duration = start_time.elapsed();
        let latency_ms = duration.as_secs_f64() * 1000.0;

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("latencies.csv")
        {
            let _ = writeln!(file, "{}", latency_ms);
        }

        let embeddings_tensor = outputs["embedded"].try_extract_tensor::<f32>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(embeddings_tensor.1.to_vec())
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
    pub fn new(model_path: &str) -> PyResult<Self> {
        println!("Candle Runtime initialized.");

        let device = Device::cuda_if_available(0)
            .map_err(|e| PyRuntimeError::new_err(format!("Device error: {}", e)))?;

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

    pub fn get_embeddings(&mut self, padded_batch: Vec<Vec<i64>>) -> PyResult<Vec<f32>> {
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

        let start_time = Instant::now();

        let (seq_emb, _attentions) = self.model.forward(&input_tensor, false)
            .map_err(|e| PyRuntimeError::new_err(format!("Forward pass error: {}", e)))?;

        let duration = start_time.elapsed();
        let latency_ms = duration.as_secs_f64() * 1000.0;

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("latencies.csv")
        {
            let _ = writeln!(file, "{}", latency_ms);
        }

        let output_vec = seq_emb
            .to_device(&Device::Cpu)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .flatten_all()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(output_vec)
    }
}