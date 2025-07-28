# BetterTherapy

## Project Setup

### 1. Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Bittensor](https://github.com/opentensor/bittensor#install)
- `btcli` (Bittensor CLI)

### 2. Install Dependencies with `uv`

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install uv if you don't have it
pip install uv

# Install all dependencies in a virtual environment
uv sync

uv pip install -e .
```

---

## Running the Subnet

### 1. Setting Up Wallets

You need wallets for the subnet owner, miner, and validator:

```bash
btcli wallet new_coldkey --wallet.name owner
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

---

## Hardware Requirements (Validator & Miner)

### Overview

This document outlines the hardware requirements for running Meta's Llama 3.1 8B model locally.

### Minimum Requirements

- **CPU**: 4 cores at 2.5 GHz
- **GPU**:
  - **Full precision (FP16)**: 16GB VRAM minimum
  - **4-bit quantized**: 6GB VRAM minimum
  - **Examples**: RTX 3060 12GB, RTX 4060 Ti 16GB
- **Memory**: 16GB RAM
- **Storage**: 20GB free space
- **OS**: Ubuntu 20.04 or later, Windows 10/11, MacOS

### Recommended Requirements

- **CPU**: 8 cores at 3.5 GHz
- **GPU**:
  - **For fast inference**: 16-24GB VRAM
  - **For fine-tuning (4-bit)**: 15GB+ VRAM
  - **Examples**: RTX 3090 (24GB), RTX 4070 Ti (16GB), RTX 4080 (16GB)
- **Memory**: 32GB RAM
- **Storage**: 50GB SSD
- **OS**: Ubuntu 22.04 or later

### VRAM Requirements by Precision

| Precision   | VRAM Required | Use Case                               |
| ----------- | ------------- | -------------------------------------- |
| FP16 (Half) | 16GB          | Full model performance                 |
| INT8        | ~10-12GB      | Good balance of performance and memory |
| 4-bit (Q4)  | 6-8GB         | Budget-friendly, slight quality loss   |

### Compatible GPUs

#### Budget Options (4-bit quantization)

- NVIDIA RTX 3060 (12GB)
- NVIDIA RTX 3070 (8GB)
- NVIDIA RTX 4060 Ti (8GB/16GB)

#### Recommended GPUs

- NVIDIA RTX 3090 (24GB)
- NVIDIA RTX 4070 Ti (16GB)
- NVIDIA RTX 4080 (16GB)
- NVIDIA RTX 4090 (24GB)

#### Professional GPUs

- NVIDIA A100 (40GB/80GB)
- NVIDIA A6000 (48GB)
- NVIDIA H100 (80GB)

### Software Requirements

- **CUDA**: Version 11.8 or higher
- **CUDA Compute Capability**: 6.0 minimum (GTX 10 series and newer)
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher with CUDA support

### Performance Notes

- Any 24GB GPU can run Llama 3.1 8B Q4K_M quantized models with 128k context
- 4-bit quantization reduces VRAM requirements to ~5-6GB for basic inference
- For Ollama deployment: 8+ CPU cores and 8GB+ VRAM recommended
- Consumer GPUs (RTX 30/40 series) provide excellent performance for this model size

### Context Length Considerations

| Quantization | VRAM | Max Context Length |
| ------------ | ---- | ------------------ |
| Q4K_M        | 24GB | 128k tokens        |
| Q8           | 24GB | 64k tokens         |
| FP16         | 24GB | 32k tokens         |

## Use Cloud Model (Miner)

1. **Cooling**: Miner can modify code to use api key models lik openai, claude, etc.

---

## Running the Miner

To start a miner (after setting up wallets and registering on your subnet):

```bash
uv run python neurons/miner.py \
  --netuid <your_netuid> \
  --subtensor.network <network> \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

- Replace `<your_netuid>` with your subnet ID (e.g., `354` (test) and `10).
- Replace `<network>` with your chain endpoint (e.g., `test` for local, or use `finney` for mainnet).

---

## Running the Validator

### Weights & Biases (wandb) Setup

This project uses [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and visualization.

1. **Sign up at [wandb.ai](https://wandb.ai/) and get your API key.**
2. **Login in your terminal:**
   ```bash
   wandb login
   ```
3. **(Optional) Set environment variables for headless runs:**
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

To start a validator (after setting up wallets and registering on your subnet):

The validator will automatically create and manage runs, groups, and dashboards in wandb. See `BetterTherapy/utils/wandb.py` for advanced usage.

```bash
uv run neurons/validator.py \
  --netuid <your_netuid> \
  --subtensor.chain_endpoint <endpoint> \
  --wallet.name validator \
  --wallet.hotkey default \
  --logging.debug
```

- Replace `<your_netuid>` with your subnet ID (e.g., `354` (test) and `10).
- Replace `<network>` with your chain endpoint (e.g., `test` for local, or use `finney` for mainnet).

The validator will automatically log evaluation metrics and charts to wandb.

---

## Local Development

- Use `uv` for all dependency management.
- All dependencies are listed in `pyproject.toml`.

---

## Additional Resources

- [docs/running_on_staging.md](docs/running_on_staging.md) — Local chain setup
- [docs/running_on_testnet.md](docs/running_on_testnet.md) — Testnet setup
- [docs/running_on_mainnet.md](docs/running_on_mainnet.md) — Mainnet setup

---
