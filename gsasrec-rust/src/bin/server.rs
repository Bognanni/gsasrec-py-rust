// Commands to start the server based on build context:
// 1. CPU-only build (standard):
//    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin server -- --device cpu --engine onnx
//
// 2. GPU-enabled build (when enabling --features cuda):
//    RUSTFLAGS="-C target-cpu=native" cargo run --release --bin server --features cuda -- --device cuda --engine onnx

use actix_web::{middleware::Logger, web, App, HttpResponse, HttpServer, Responder};
use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor};
use candle_nn::VarBuilder;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use serde::{Deserialize, Serialize};
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
// async channel for sending inference results back to the request handler
use tokio::sync::oneshot;

#[cfg(feature = "cuda")]
use ort::ep::{self, ExecutionProvider};

use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::model::GSASRec;

// mimalloc useful to reduce memory fragmentation and improve performance for large batch processing
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Deserialize)]
struct EmbeddingsRequest {
    batch_sequences: Vec<Vec<u32>>,
}

// light weight serializer to avoid unnecessary copies of the embeddings
// a' is a lifetime parameter that allows us to borrow slices of the embeddings without taking ownership
#[derive(Serialize)]
struct SliceSerializer<'a> {
    user_embeddings: Vec<&'a [f32]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EngineType {
    Candle,
    Onnx,
}

struct Args {
    use_cuda: bool,
    engine: EngineType,
}

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().collect();
        let mut use_cuda = false; // Default su CPU come richiesto
        let mut engine = EngineType::Onnx;

        let mut i = 1;
        while i < raw.len() {
            if raw[i] == "--device" && i + 1 < raw.len() {
                use_cuda = raw[i + 1].trim().to_lowercase() == "cuda";
                i += 2;
            } else if raw[i] == "--engine" && i + 1 < raw.len() {
                let eng_str = raw[i + 1].trim().to_lowercase();
                if eng_str == "onnx" {
                    engine = EngineType::Onnx;
                } else if eng_str == "candle" {
                    engine = EngineType::Candle;
                }
                i += 2;
            } else {
                i += 1;
            }
        }
        Self { use_cuda, engine }
    }
}

enum ModelBackend {
    // Arc implemented to allow sharing the model across threads safely
    Candle(Arc<GSASRec>),
    // Mutex implemented because it needs to be mutable for ONNX inference, and Arc allows sharing across threads
    Onnx(Arc<Mutex<Session>>),
}

impl Clone for ModelBackend {
    fn clone(&self) -> Self {
        match self {
            ModelBackend::Candle(m) => ModelBackend::Candle(Arc::clone(m)),
            ModelBackend::Onnx(s) => ModelBackend::Onnx(Arc::clone(s)),
        }
    }
}

// function to clip sequences to a maximum length, returning a slice of the original sequence
fn clip_sequence(seq: &[u32], max_len: usize) -> &[u32] {
    if seq.len() > max_len {
        &seq[(seq.len() - max_len)..]
    } else {
        seq
    }
}

// function that finds the maximum sequence length in a batch
fn prepare_dynamic_batch(sequences: &[Vec<u32>], max_supported_len: usize, pad_val: u32,)
-> (Vec<i64>, Vec<u32>, usize, usize) {
    let b = sequences.len();
    if b == 0 {
        return (vec![], vec![], 0, 0);
    }

    // max length in the batch
    let target_len = sequences
        .iter()
        .map(|seq| seq.len().min(max_supported_len))
        .max()
        .unwrap_or(1);

    // flat 1d vectors to hold the padded sequences, pre-allocated to avoid multiple heap allocations
    let mut flat_i64 = Vec::with_capacity(b * target_len);
    let mut flat_u32 = Vec::with_capacity(b * target_len);

    for seq in sequences {
        let clipped = clip_sequence(seq, max_supported_len);
        let pad_size = target_len.saturating_sub(clipped.len());

        // left padding with pad_val to ensure all sequences are of the same length
        for _ in 0..pad_size {
            flat_i64.push(pad_val as i64);
            flat_u32.push(pad_val);
        }
        // append the real values of the sequence
        for &val in clipped {
            flat_i64.push(val as i64);
            flat_u32.push(val);
        }
    }

    (flat_i64, flat_u32, b, target_len)
}


fn run_inference(model: &ModelBackend, device: &CandleDevice, max_seq_len: usize, pad_val: u32, sequences: Vec<Vec<u32>>,)
-> Result<(Vec<f32>, usize), String> {
    // compute the dynamic shape [batch_size, target_len]
    let (flat_i64, flat_u32, b, target_len) = prepare_dynamic_batch(&sequences, max_seq_len, pad_val);
    
    if b == 0 {
        return Ok((vec![], 0));
    }

    match model {
        ModelBackend::Onnx(session_mutex) => {
            let input_shape = vec![b, target_len];
            let input_tensor = Value::from_array((input_shape, flat_i64))
                .map_err(|e| format!("ORT Value error: {}", e))?;

            let inputs = ort::inputs!["input_seq" => input_tensor];
            let mut session_lock = session_mutex.lock().map_err(|_| "Mutex poisoned".to_string())?;

            let outputs = session_lock.run(inputs).map_err(|e| format!("ONNX run error: {}", e))?;
            let extracted = outputs["embedded"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;

            let shape = &extracted.0;
            let data = &extracted.1;
            let out_seq_len = shape[1] as usize;
            let hidden_dim = shape[2] as usize;

            let mut flat_result = Vec::with_capacity(b * hidden_dim);
            for batch_idx in 0..b {
                // take the last token embedding for each sequence in the batch
                let start_idx = (batch_idx * out_seq_len + (out_seq_len - 1)) * hidden_dim;
                let end_idx = start_idx + hidden_dim;
                flat_result.extend_from_slice(&data[start_idx..end_idx]);
            }

            Ok((flat_result, hidden_dim))
        }
        ModelBackend::Candle(model) => {
            let input_tensor = Tensor::from_vec(flat_u32, (b, target_len), device)
                .map_err(|e| format!("Candle tensor error: {}", e))?;

            let (seq_emb, _) = model.forward(&input_tensor, false).map_err(|e| e.to_string())?;
            let last_token = seq_emb.i((.., target_len - 1, ..)).map_err(|e| e.to_string())?;
            let hidden_dim = last_token.dims()[1];

            let flat_embeddings = last_token
                .to_device(&CandleDevice::Cpu).map_err(|e| e.to_string())?
                .flatten_all().map_err(|e| e.to_string())?
                .to_vec1::<f32>().map_err(|e| e.to_string())?;

            Ok((flat_embeddings, hidden_dim))
        }
    }
}

struct BatchItem {
    sequences: Vec<Vec<u32>>,
    // the batcher will send the result back to the request handler through this oneshot channel
    // the value sent back is a Result containing either the serialized embeddings or an error message
    responder: oneshot::Sender<Result<Vec<u8>, String>>,
}

struct DynamicBatcher {
    // the sender side of the channel used to send BatchItems to the batcher thread
    sender: std_mpsc::Sender<BatchItem>,
}

impl DynamicBatcher {
    fn new(model: ModelBackend, device: CandleDevice, seq_len: usize, pad_val: u32) -> Self {
        let max_batch_size: usize = std::env::var("GSASREC_MAX_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);

        let max_wait = Duration::from_secs_f64(
            std::env::var("GSASREC_MAX_WAIT_MS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(1.0)
                / 1000.0,
        );

        let (tx, rx) = std_mpsc::channel::<BatchItem>();

        // separate thread for batching and inference to avoid blocking the async runtime
        std::thread::spawn(move || {
            loop {
                let first_item = match rx.recv() {
                    Ok(item) => item,
                    Err(_) => return,
                };

                let mut batch_items = vec![first_item];
                let deadline = Instant::now() + max_wait;

                while batch_items.len() < max_batch_size {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match rx.recv_timeout(remaining) {
                        Ok(item) => batch_items.push(item),
                        Err(_) => break,
                    }
                }

                let item_sizes: Vec<usize> = batch_items.iter().map(|it| it.sequences.len()).collect();
                let merged: Vec<Vec<u32>> = batch_items
                    .iter_mut()
                    .flat_map(|it| std::mem::take(&mut it.sequences))
                    .collect();

                let result = run_inference(&model, &device, seq_len, pad_val, merged);

                match result {
                    Ok((flat_embeddings, hidden_dim)) => {
                        let mut offset = 0;
                        for (item, size) in batch_items.into_iter().zip(item_sizes.into_iter()) {
                            let mut user_slices = Vec::with_capacity(size);
                            for _ in 0..size {
                                let slice = &flat_embeddings[offset..offset + hidden_dim];
                                user_slices.push(slice);
                                offset += hidden_dim;
                            }

                            let payload_wrapper = SliceSerializer {
                                user_embeddings: user_slices,
                            };
                            let json_bytes = serde_json::to_vec(&payload_wrapper)
                                .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e).into_bytes());

                            let _ = item.responder.send(Ok(json_bytes));
                        }
                    }
                    Err(err) => {
                        for item in batch_items {
                            let _ = item.responder.send(Err(err.clone()));
                        }
                    }
                }
            }
        });

        DynamicBatcher { sender: tx }
    }

    async fn submit(&self, sequences: Vec<Vec<u32>>) -> Result<Vec<u8>, String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let item = BatchItem {
            sequences,
            responder: resp_tx,
        };
        self.sender
            .send(item)
            .map_err(|_| "Batcher task is not running".to_string())?;
        resp_rx
            .await
            .map_err(|_| "Batcher dropped the response channel".to_string())?
    }
}

struct AppState {
    batcher: DynamicBatcher,
}

async fn embeddings_endpoint(state: web::Data<AppState>, req: web::Json<EmbeddingsRequest>,) -> impl Responder {
    if req.batch_sequences.is_empty() {
        return HttpResponse::Ok()
            .content_type("application/json")
            .body("{\"user_embeddings\":[]}");
    }

    match state.batcher.submit(req.batch_sequences.clone()).await {
        Ok(json_bytes) => HttpResponse::Ok()
            .content_type("application/json")
            .body(json_bytes),
        Err(err) => HttpResponse::InternalServerError().body(err),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    log::info!("Starting inference server...");
    let args = Args::parse();

    let num_items = 3416;
    let sequence_length = 200;
    let pad_val = 0;

    let device = if args.use_cuda {
        #[cfg(feature = "cuda")]
        {
            CandleDevice::cuda_if_available(0).expect("Failed to initialize CUDA")
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!("CUDA fallback to CPU.");
            CandleDevice::Cpu
        }
    } else {
        CandleDevice::Cpu
    };

    let model_backend = match args.engine {
        EngineType::Candle => {
            let model_path = "model.safetensors";
            log::info!("Loading Candle backend...");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                    .expect("Failed to read safetensors")
            };
            let mut config = GsasrecConfig::new("infer", num_items as u32);
            config.sequence_length = sequence_length;
            let model = GSASRec::new(vb, config).expect("Failed to initialize GSASRec");
            ModelBackend::Candle(Arc::new(model))
        }
        EngineType::Onnx => {
            let model_path = "models/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962_wrapped_embeddings.onnx";
            log::info!("Loading ONNX backend...");
            let _ = ort::init().with_name("gsasrec_inference").commit();

            let mut builder = Session::builder()
                .expect("Failed to create Session builder")
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .expect("Failed to set optimization level")
                .with_intra_threads(1)
                .expect("Failed to set intra_threads")
                .with_inter_threads(1)
                .expect("Failed to set inter_threads");

            #[cfg(feature = "cuda")]
            if args.use_cuda {
                let cuda = ep::CUDA::default();
                if cuda.is_available().unwrap_or(false) {
                    cuda.register(&mut builder).expect("Failed to register CUDA");
                }
            }

            let session = builder
                .commit_from_file(model_path)
                .expect("Failed to load ONNX model");
            ModelBackend::Onnx(Arc::new(Mutex::new(session)))
        }
    };

    let bind_address = "0.0.0.0:8080";
    log::info!("Server listening at http://{}", bind_address);

    let batcher = DynamicBatcher::new(model_backend.clone(), device.clone(), sequence_length, pad_val);
    
    let app_state = web::Data::new(AppState {
        batcher
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(app_state.clone())
            .route("/get_embeddings/pure_rust", web::post().to(embeddings_endpoint))
    })
    .workers(1)
    .bind(bind_address)?
    .run()
    .await
}