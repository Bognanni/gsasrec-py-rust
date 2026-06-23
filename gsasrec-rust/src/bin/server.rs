// command to do the load test: (-q is the rps, -z is the duration of the test, change anytime)
// oha -m POST -T application/json -D payload.json -q 50 -z 60s http://127.0.0.1:9090/embeddings
// to do the warm up before each test:
// oha -m POST -T application/json -D payload.json -n 30 -c 10 http://127.0.0.1:9090/embeddings > /dev/null
// command to run the server if we use the CPU:
// RUSTFLAGS="-C target-cpu=native" cargo run --release --bin server -- --device cpu
// to call the CUDA server:
// cargo run --release --bin server -- --device cuda
// to kill processes using the port:
// fuser -k 9090/tcp and then kill -9 <PID>

use actix_web::{web, App, HttpServer, HttpResponse, Responder, middleware::Logger};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::model::GSASRec;


#[derive(Deserialize)]
struct EmbeddingsRequest {
    sequences: Vec<Vec<u32>>, 
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    user_embeddings: Vec<Vec<Vec<f32>>>, 
    time_ms: f32,
}

struct Args {
    use_cuda: bool,
}

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().collect();

        let mut use_cuda = true; 

        let mut i = 1;
        while i < raw.len() {
            if raw[i] == "--device" && i + 1 < raw.len() {
                use_cuda = raw[i+1].trim().to_lowercase() == "cuda";
                i += 2;
            } else {
                i += 1;
            }
        }

        Self { use_cuda }
    }
}

struct AppState {
    model: Arc<GSASRec>,
    device: Device,
    sequence_length: usize,
    pad_val: u32,
}


async fn embeddings_endpoint(
    state: web::Data<AppState>,
    req: web::Json<EmbeddingsRequest>,
) -> impl Responder {
    let t0 = Instant::now();
    
    let sequences = req.sequences.clone();
    
    let model = Arc::clone(&state.model);
    let device = state.device.clone();
    let seq_len = state.sequence_length;
    let pad_val = state.pad_val;

    let infer_result = web::block(move || {
        let b = sequences.len();
        if b == 0 {
            return Ok::<Vec<Vec<Vec<f32>>>, String>(vec![]);
        }

        let mut flat: Vec<u32> = Vec::with_capacity(b * seq_len);

        for seq in sequences.iter() {
            let mut inp = seq.clone();
            if inp.len() > seq_len {
                inp = inp[(inp.len() - seq_len)..].to_vec();
            } else if inp.len() < seq_len {
                let diff = seq_len - inp.len();
                let mut padded = vec![pad_val; diff];
                padded.extend_from_slice(&inp);
                inp = padded;
            }
            flat.extend_from_slice(&inp);
        }

        let input_tensor = Tensor::from_vec(flat, (b, seq_len), &device)
            .map_err(|e| format!("Error creating tensor: {}", e))?;

        let (seq_emb, _attentions) = model.forward(&input_tensor, false)
            .map_err(|e| format!("Error in forward pass: {}", e))?;

        let embeddings_vec: Vec<Vec<Vec<f32>>> = seq_emb.to_vec3()
            .map_err(|e| format!("Error to_vec3: {}", e))?;

        Ok(embeddings_vec)
    }).await;
    
    match infer_result {
        Ok(Ok(embs)) => {
            HttpResponse::Ok().json(EmbeddingsResponse {
                user_embeddings: embs,
                time_ms: t0.elapsed().as_secs_f32() * 1000.0,
            })
        },
        Ok(Err(err)) => {
            log::error!("Error in inference: {}", err);
            HttpResponse::InternalServerError().body(err)
        },
        Err(err) => {
            log::error!("Error in worker thread: {}", err);
            HttpResponse::InternalServerError().body("Critical server error")
        },
    }
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    log::info!("Starting the server...");
    
    let args = Args::parse();

    let device = if args.use_cuda {
        log::info!("CUDA requested (or default). Initializing GPU (falling back to CPU if unavailable).");
        Device::cuda_if_available(0)
            .expect("Critical error: Unable to initialize CUDA/CPU device")
    } else {
        log::info!("CPU explicitly requested by user.");
        Device::Cpu
    };

    log::info!("Device in use: {:?}", device);

    let model_path = "model.safetensors"; 
    let num_items = 3416;
    let sequence_length = 200;

    log::info!("Reading weights from {}...", model_path);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
            .expect("Critical error: Unable to read safetensors file")
    };

    let mut config = GsasrecConfig::new("infer", num_items as u32);
    config.sequence_length = sequence_length;

    let model = GSASRec::new(vb, config).expect("Critical error: Unable to create model");
    
    let app_state = web::Data::new(AppState {
        model: Arc::new(model),
        device,
        sequence_length,
        pad_val: num_items as u32 + 1,
    });

    let bind_address = "0.0.0.0:9090";
    log::info!("Model loaded! Server listening on http://{}", bind_address);

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(app_state.clone())
            .route("/embeddings", web::post().to(embeddings_endpoint))
    })
    .bind(bind_address)?
    .run()
    .await
}