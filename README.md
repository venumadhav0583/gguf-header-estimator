# gguf-header-estimator

Code and data for "A Header-Only, Attention-Aware Memory Estimator for GGUF-Quantized LLMs on Edge Devices" (V. Kalluru, IEEE AIoT 2026, short paper 161, camera-ready).

The estimator reads the leading architecture fields of a GGUF file (one 4 KB HTTP Range request for every model tested) and predicts llama.cpp peak resident memory on a CPU host with `--no-mmap`:

    M = M_w + M_KV(C) + M_rt + M_att(C)

## Quick start

    python3 gguf_header_estimator.py https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf 8192 f16

Standard library only (Python 3.9+). The output contains the parsed fields, the bytes transferred, a reserved-capacity estimate (short prompt) and a filled-context estimate.
The coefficients are calibrated on the paper's host (2 vCPU Xeon, 8 GB RAM). Refit them for any other backend.

## Layout

| Path | Contents |
|---|---|
| `gguf_header_estimator.py` | Reference parser and estimator used in the camera-ready |
| `coefficients_filled.json` | Experiment 2 in-sample gamma and beta |
| `e1_original/` | Experiment 1 as submitted: harness (`run_eval_grid.py`, `measure_peak_rss.py`), the 36-configuration measurements, the submitted estimator (`submitted_estimator.py`, kept unchanged for comparison) |
| `run_grid2.py`, `run_rss.py`, `fill_corpus.txt`, `prompts/` | Experiment 2 harness: short-prompt replication, filled-context runs, mmap runs |
| `results_grid2.jsonl`, `logs/`, `grid2.out` | Experiment 2 raw results and full llama.cpp logs |
| `analysis/` | Re-analysis of E1 (LOMO, cluster bootstrap, ablation, decision margin), gguf-parser comparison, header audit, remote range test, E2 analysis, and the script that writes every number in the paper (`final_numbers.py`, `make_e2numbers.py`) |
| `paper/` | Camera-ready LaTeX, generated number macros, PDF |
| `models_sha256.txt` | SHA-256 of the six GGUF files (download with `dl.sh`) |
| `SHA256SUMS` | Checksums of every file in this repository |

## Reproducing

1. `bash dl.sh` downloads the six Q4_K_M checkpoints; compare against `models_sha256.txt`.
2. `bash setup.sh` builds llama.cpp CPU-only. Experiment 2 used commit `339f324` (see `llama_commit.txt`); clone without `--depth 1` and `git checkout 339f324` to match it. Experiment 1 used build `b1-050dde5`.
3. `python3 run_grid2.py` reruns Experiment 2 (about 1 hour on 2 vCPUs; Phi-3 at 8K filled needs about 6 GiB free).
4. From the repository root: `python3 analysis/reanalyze.py`, `python3 analysis/analyze_grid2.py`, `python3 analysis/final_numbers.py`, `python3 analysis/make_e2numbers.py`.
5. gguf-parser comparison: `analysis/run_ggufparser.py` (gguf-parser v0.26.4 binary required).

## Scope

CPU only, one host class, models of 0.5-3.8 B parameters, llama.cpp with flash attention off. No GPU, Jetson or unified-memory measurements. mmap loading is measured but not modeled.

## License

MIT
