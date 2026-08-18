# Compute Environment Details

This file specifies the exact hardware and software environment used
for all experiments reported in the AdverSentinel-LLM paper (CMC 2025).

---

## Training Environment

| Component          | Specification                                   |
|--------------------|-------------------------------------------------|
| GPU                | 4× NVIDIA A100 SXM4 80GB                        |
| GPU Driver         | 525.89.02                                       |
| CUDA Version       | 12.1                                            |
| cuDNN Version      | 8.9.2                                           |
| CPU                | Intel Xeon Platinum 8380 (2× socket, 40 cores)  |
| RAM                | 512 GB DDR4 ECC                                 |
| Storage            | 2 TB NVMe SSD                                   |
| Interconnect       | NVLink 3.0 (600 GB/s bidirectional per GPU pair)|
| OS                 | Ubuntu 22.04.3 LTS                              |
| Python             | 3.10.12                                         |
| PyTorch            | 2.1.0+cu121                                     |
| Transformers       | 4.38.2                                          |
| Precision          | FP16 (mixed-precision, torch.cuda.amp)          |
| Multi-GPU strategy | DataParallel (nn.DataParallel)                  |

**Training Duration:** ~6 hours (50 epochs, batch size 128, 4× A100)

---

## Inference / Latency Benchmarking Environment

Latency numbers reported in Table 9 were measured on the following hardware
configurations, each run independently with a warm-up of 100 prompts before
timing:

| Hardware                     | Driver  | CUDA | GPU Mem | Mean Lat | Throughput  |
|------------------------------|---------|------|---------|----------|-------------|
| NVIDIA A100 SXM4 80GB        | 525.89  | 12.1 | 2.4 GB  | 8.7 ms   | 1,142 p/s   |
| NVIDIA V100 SXM2 32GB        | 520.61  | 11.8 | 2.4 GB  | 12.1 ms  | 823 p/s     |
| NVIDIA T4 16GB               | 525.89  | 12.1 | 2.3 GB  | 14.3 ms  | 697 p/s     |
| NVIDIA RTX 3090 24GB         | 535.54  | 12.2 | 2.4 GB  | 11.8 ms  | 845 p/s     |
| Intel Xeon 8380 (CPU only)   | N/A     | N/A  | N/A     | 67.4 ms  | 148 p/s     |
| NVIDIA Jetson Orin NX (INT4) | JetPack 6.0 | 11.4 | 0.8 GB | 4.8 ms | 208 p/s  |

**Latency measurement protocol:**
- Input: single prompt, max_length=512 tokens
- Includes: tokenization + TFE + dual-encoder forward + DTMB search + AAGM
- Excludes: model loading, first-batch JIT compilation
- Averaged over 1,000 forward passes after 100-prompt warm-up
- `torch.cuda.synchronize()` called before stopping the timer

---

## Quantization Configurations (Table 9)

| Configuration          | Method               | GPU Mem | Latency | F1    | ZDR    |
|------------------------|----------------------|---------|---------|-------|--------|
| FP16 (default)         | PyTorch AMP          | 1.9 GB  | 11.8 ms | 0.967 | 94.8%  |
| INT8 post-training     | torch.quantization   | 1.4 GB  | 9.2 ms  | 0.964 | 94.1%  |
| INT4 (GPTQ)            | AutoGPTQ             | 0.8 GB  | 4.3 ms  | 0.943 | 90.1%  |
| DistilCDET (4-layer KD)| Knowledge distill.   | ~0.5 GB | 6.1 ms  | 0.951 | 91.3%  |

---

## Multi-Tenant Co-Deployment (Table 9)

Simulated by co-deploying AdverSentinel-LLM alongside a quantized
LLaMA-3-8B (INT8) model on a single NVIDIA A100-80GB GPU,
with 50% GPU memory reserved for the LLM backend.

| Metric            | Value               |
|-------------------|---------------------|
| Mean latency      | 18.7 ms             |
| p99 latency       | 23.4 ms             |
| Latency increase  | +30.8% vs dedicated |
| LLM model         | LLaMA-3-8B INT8     |
| Memory split      | ~40 GB / 80 GB      |

---

## Software Package Versions (Full)

```
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
transformers==4.38.2
datasets==2.18.0
tokenizers==0.15.2
accelerate==0.27.2
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.4.1
pyyaml==6.0.1
tqdm==4.66.2
einops==0.7.0
tensorboard==2.16.2
pandas==2.2.1
```

To reproduce the exact software environment:
```bash
conda create -n adversentinel python=3.10.12 -y
conda activate adversentinel
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
