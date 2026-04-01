# 🚀 MLLM: A Multimodal Time Series Foundation Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Dependency Manager: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**MLLM** (Multimodal Large Language Model for Time Series) is a state-of-the-art foundation model designed to bridge the gap between numerical time series forecasting and natural language semantics. By leveraging pre-trained LLMs (GPT-2, LLaMA, BERT) alongside specialized temporal layers, MLLM captures complex patterns and cross-modal correlations for superior forecasting performance.

---

## 🌟 Key Features

- **Multimodal Integration**: Seamlessly combines time series data with textual prompts and embeddings.
- **Foundation Model Backbones**: Supports multiple LLM architectures including GPT-2, LLaMA-2/3, and BERT.
- **Advanced Temporal Layers**: Includes implementations for AutoCorrelation, Fourier correlation, and Multi-Wavelet mechanisms.
- **Fast Development**: Battery-included setup with `uv` for lightning-fast dependency management and `just` for task automation.
- **Extensive Benchmarking**: Pre-configured scripts for various real-world datasets (Agriculture, Energy, Health, etc.).

---

## 🛠️ Quick Start

### 1. Environment Setup
We use [uv](https://github.com/astral-sh/uv) for high-performance dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd MLLM

# Install all dependencies and setup virtual environment
uv sync --all-groups
```

### 2. Prepare Data & Embeddings
Before running multimodal experiments, you need to generate text embeddings for your datasets:

```bash
# Generate text-embeddings using the provided script
just sh scripts/gen_text_embedding.sh
```
> [!TIP]
> Use the `generate_embedding.py` script directly for more granular control over embedding models and dimensions.

### 3. Run Experiments
You can run individual experiments using `run.py` or execute full benchmark suites.

**Single Experiment:**
```bash
just run run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id MLLM_Test \
  --model Autoformer \
  --data Agriculture \
  --seq_len 96 \
  --pred_len 96 \
  --llm_model GPT2
```

**Run Full Benchmark Suite:**
```bash
just sh scripts/forecasting/timecma.sh
```

---

## 📁 Project Structure

```text
├── data/               # Raw datasets (excluded from git)
├── data_provider/      # Data factories and loaders (Multimodal & Unimodal)
├── exp/                # Experiment management (training/validation/testing loops)
├── layers/             # Neural network building blocks (Attention, Conv, etc.)
├── llm/                # LLM integration logic (Prompting & Embedding)
├── scripts/            # Automation shell scripts for large-scale runs
├── utils/              # Metrics, logging, and helper utilities
├── run.py              # Main entry point for multimodal models
├── run_unimodal.py     # Entry point for unimodal baseline models
├── justfile            # Task automation commands
└── pyproject.toml      # Project configuration and dependencies
```

---

## 🏎️ Common Commands (`just`)

This project uses `just` as a command runner. Typical workflows:

| Command | Description |
| :--- | :--- |
| `just run <script>` | Execute a python script within the `uv` environment |
| `just sh <script>` | Run a bash script (e.g., benchmark loops) |
| `just notebook` | Launch Jupyter Lab for analysis |
| `just lint` | Automatically fix and format code using `ruff` |
| `just sync` | Synchronize dependencies with the lockfile |
| `just clean` | Remove temporary cache files |

---

## 📊 Experiment Tracking
MLLM supports integrated experiment tracking via:
- **MLflow**: Enable with `--tracking_mlflow 1`
- **TensorBoard**: Logs are stored in `./logs/` by default.

---

## 📜 Citation

If you use MLLM in your research, please cite our repository:

```bibtex
@misc{mllm2026,
  author = {AISEED},
  title = {MLLM: A Multimodal Time Series Foundation Model},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aiseed/MLLM}}
}
```