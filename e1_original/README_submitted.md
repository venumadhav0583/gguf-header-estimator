# Reproducibility Bundle — AIoT 2026 Submission

## Contents
- `gguf_header_estimator.py` — GGUF v3 header parser + estimator (241 LOC, dependency-free Python).
- `measure_peak_rss.py` — measurement harness: launches llama-cli, samples /proc/PID/status VmRSS at 100 ms, records peak.
- `run_eval_grid.py` — runs the full 36-configuration grid (6 models × 2 KV formats × 3 context lengths, 3 trials each). ~30–90 min on 2 vCPU.
- `analyze.py` — computes MAPE (with bootstrap 95% CI), fits (alpha, beta) via OLS, runs the ablation, produces scatter data.
- `measurements_36cfg.json` — raw peak-RSS results (3 trials per config, max reported).
- `analysis_summary.json` — computed MAPE, Pearson r, RMSE, ablation, per-model characteristics, per-config residuals, bootstrap CI.

## Reproduce from scratch (CPU-only)
```bash
# 1. Install prerequisites
pip3 install --user cmake

# 2. Build llama.cpp (CPU only, portable)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF
cmake --build build --config Release -j 2 --target llama-cli
# Binary: build/bin/llama-cli

# 3. Download models (Q4_K_M)
mkdir -p models && cd models
curl -L -o qwen2-0.5b.Q4_K_M.gguf     "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf"
curl -L -o tinyllama-1.1b.Q4_K_M.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
curl -L -o llama-3.2-1b.Q4_K_M.gguf   "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
curl -L -o qwen2-1.5b.Q4_K_M.gguf     "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf"
curl -L -o gemma-2-2b.Q4_K_M.gguf     "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
curl -L -o phi-3-mini.Q4_K_M.gguf     "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

# 4. Run the grid
cd ..
python3 run_eval_grid.py   # writes results.json, ~30-90 min

# 5. Analyze
python3 analyze.py         # prints MAPE, ablation, bootstrap CI; writes analysis.json
```

## Environment used in the paper
- Host: 2 vCPU Intel Xeon @ 2.9 GHz, 8 GB RAM, no GPU
- Linux 6.x, Python 3.11
- llama.cpp build: `b1-050dde5` (2026)
- gcc/g++ 15.2.0
- 3 trials per configuration; run-max reported

## Key results reproduced by `analyze.py`
- N = 36 configurations
- MAPE = 2.05% (bootstrap 95% CI 1.44–2.73%, n=2000)
- Pearson r = 0.9989
- RMSE = 0.049 GB
- Fitted overhead: α = 0.0009 GB/B, β = 0.077 GB
- Ablation: drop-eff-KV-heads → 31.11% (69.0% at ctx=8192)

## F-test / nested-model significance test
- `ftest.py` runs the nested-model F-test, paired Wilcoxon signed-rank, and 20k-permutation test comparing the full estimator vs. the params-only baseline on the 36-config grid. Reads `analysis.json`; writes `ftest.json`.
- Result: F(2,33) = 3429.3, p < 10^-38 (partial eta^2 = 0.9952). Wilcoxon z = -5.23, p = 1.7e-7. Permutation p = 5e-5.
