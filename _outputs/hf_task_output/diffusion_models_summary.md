# Shortlist: Recent diffusion models useful for ML engineers

1. stabilityai/stable-diffusion-3.5-large
   - Link: https://hf.co/stabilityai/stable-diffusion-3.5-large
   - Why it matters: High-quality, general-purpose text-to-image model with an official Diffusers pipeline and wide ecosystem support (playground, spaces) — good baseline for production and experimentation.

2. stabilityai/stable-diffusion-3.5-large-tensorrt
   - Link: https://hf.co/stabilityai/stable-diffusion-3.5-large-tensorrt
   - Why it matters: Official TensorRT-optimized build (ONNX + TensorRT, FP8/quantization support) for substantially faster inference on NVIDIA GPUs — practical for low-latency production deployments.

3. stabilityai/stable-diffusion-3.5-large-turbo_amdgpu
   - Link: https://hf.co/stabilityai/stable-diffusion-3.5-large-turbo_amdgpu
   - Why it matters: AMD/ROCm-targeted variant of the turbo release; useful for teams running AMD GPUs or seeking alternatives to CUDA-centric pipelines.

Comparative summary

Quality (sample fidelity): stabilityai/stable-diffusion-3.5-large is the baseline for highest perceptual fidelity among the three — it provides the best out-of-the-box image quality and is the model most research and third-party tooling target. The TensorRT and AMD turbo builds are functionally equivalent in fidelity when using the same base weights, though quantization (FP8/INT8) or aggressive optimization in the TensorRT build can introduce minor artifacts; in practice fidelity remains high for most use cases.

Inference speed: the TensorRT variant is optimized for NVIDIA data-centers and offers the largest speedups (reduced latency and higher throughput), especially when using FP16/FP8 and TensorRT engines. The turbo_amdgpu variant improves inference on ROCm-enabled AMD cards versus unoptimized Diffusers runs but typically lags a well-optimized TensorRT pipeline on NVIDIA hardware.

Compute requirements: the vanilla stabilityai/stable-diffusion-3.5-large pipeline (Diffusers) benefits from modern GPUs (e.g., A10, A100, or comparable 24+ GB cards) for comfortable 1–2s 512px generation at batch=1 with FP16. The TensorRT build reduces memory and GPU time via quantization/engine optimizations and can run acceptably on narrower GPUs (e.g., 16GB-class) depending on precision and batching. The AMD turbo build targets ROCm stacks and requires compatible AMD GPUs and drivers; memory needs are similar to the unoptimized model unless quantized.

Best use cases: use stabilityai/stable-diffusion-3.5-large as the development and quality reference; deploy stabilityai/stable-diffusion-3.5-large-tensorrt for production inference on NVIDIA servers where latency and throughput matter; choose stabilityai/stable-diffusion-3.5-large-turbo_amdgpu if your infra is AMD/ROCm-based or you need an alternative to CUDA.

Notes: These are official StabilityAI model cards/repos on the Hub. The TensorRT and AMD variants reference or are derived from the large/turbo base; consult each repo's README for exact installation, hardware, and quantization steps (some gated access applies).