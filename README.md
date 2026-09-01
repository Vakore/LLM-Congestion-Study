# LLM Inference Traffic Characterization for Congestion Control
### M.S. Computer Science, William & Mary - Fall 2026

This repository contains all scripts, instructions, and analysis code for the
research project: *Characterizing LLM Inference Traffic for Congestion Control:
A Measurement and Gap Analysis Study.*

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Installing CUDA](#3-installing-cuda)
4. [Installing Python Dependencies](#4-installing-python-dependencies)
5. [Downloading the Model](#5-downloading-the-model)
6. [Downloading the Dataset](#6-downloading-the-dataset)
7. [Running the vLLM Server](#7-running-the-vllm-server)
8. [Traffic Capture Setup](#8-traffic-capture-setup)
9. [Running Experiments](#9-running-experiments)
10. [Analyzing Results](#10-analyzing-results)
11. [Directory Structure](#11-directory-structure)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Requirements

| Component | Minimum | Your Hardware |
|---|---|---|
| GPU | 8 GB VRAM | ASUS ROG Zephyrus G14 (RTX 4060/4070 Mobile) |
| RAM | 16 GB | 16–32 GB |
| Disk | 30 GB free | — |
| OS | Ubuntu 22.04 LTS | Native or WSL2 (see note below) |
| CUDA | 12.1+ | — |
| Python | 3.10–3.12 | — |

> **WSL2 Note:** If you are on Windows, use WSL2 with Ubuntu 22.04. GPU
> passthrough works natively with WSL2 + recent NVIDIA drivers (≥535). Do NOT
> use WSL1. Open the Microsoft Store, install "Ubuntu 22.04 LTS", then follow
> all instructions below inside that WSL2 terminal.
>
> **Native Linux Note:** If you dual-boot or run Ubuntu natively, skip the WSL2
> steps and proceed directly to Section 2.

---

## 2. Environment Setup

### 2a. If Using WSL2 (Windows)

Install WSL2 and Ubuntu 22.04 if you have not already:

```powershell
# Run in Windows PowerShell (as Administrator)
wsl --install -d Ubuntu-22.04
```

```
P-NOTE:
Distribution successfully installed. It can be launched via 'wsl.exe -d Ubuntu-22.04'
Launching Ubuntu-22.04...
Provisioning the new WSL instance Ubuntu-22.04
This might take a while...
Create a default Unix user account: default
New password:
Retype new password:
passwd: password updated successfully
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.

default@User:/mnt/c/WINDOWS/system32$

default, default
```

Restart your machine. Then open the Ubuntu 22.04 app and set up your username
and password. All subsequent commands run inside this Ubuntu terminal.

Install the NVIDIA CUDA drivers for WSL2. Download from:

> https://developer.nvidia.com/cuda-downloads
>
> Select: Linux → x86_64 → WSL-Ubuntu → 2.0 → deb (network)

Follow the installation commands shown on that page exactly. They will look like:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

After installing, verify:

```bash
nvidia-smi
# Should show your RTX GPU and CUDA version
```

### 2b. If Using Native Linux (Ubuntu 22.04)

Install CUDA using the same download page above, selecting:

> Linux → x86_64 → Ubuntu → 22.04 → deb (network)

Then run the provided commands and verify with `nvidia-smi`.

### 2c. System Packages (Both)

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    tcpdump wireshark-common tshark \
    git curl wget build-essential \
    libpcap-dev net-tools
```

When prompted about letting non-superusers capture packets, select **Yes**.
Then add yourself to the wireshark group:

```bash
sudo usermod -aG wireshark $USER
newgrp wireshark
```

---

## 3. Installing CUDA (Verification)

After completing Section 2, confirm CUDA is accessible:

```bash
nvcc --version
# Expected: Cuda compilation tools, release 12.x

python3 -c "import subprocess; subprocess.run(['nvidia-smi'])"
```

If `nvcc` is not found, add CUDA to your PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Installing Python Dependencies

Clone this repository and set up the virtual environment:

```bash
git clone <your-repo-url> llm-inference-traffic
cd llm-inference-traffic

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Important:** Always activate the virtual environment before running any
> script in this project:
> ```bash
> source .venv/bin/activate
> ```

vLLM installation can take 10–15 minutes and downloads several GB of wheels.
If the install fails, see [Troubleshooting](#12-troubleshooting).

---

## 5. Downloading the Model

You will use a quantized model to fit within the G14's VRAM. The recommended
model is **Mistral 7B Instruct v0.3 AWQ**, which requires approximately 5–6 GB
VRAM and runs at full quality for this task.

### 5a. Create a HuggingFace Account

Go to: https://huggingface.co/join

Create a free account. No paid plan is needed.

### 5b. Accept Model License (Mistral)

Navigate to:
> https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

Click **"Agree and access repository"** to accept the license.

For the AWQ-quantized version (recommended):
> https://huggingface.co/solidrust/Mistral-7B-Instruct-v0.3-AWQ

No separate license acceptance needed for this quantized derivative.

### 5c. Get Your HuggingFace Token

Go to: https://huggingface.co/settings/tokens

Click **"New token"** → name it `research` → Role: **Read** → Generate.
Copy the token (starts with `hf_...`).

### 5d. Download the Model

```bash
# Activate your venv first
source .venv/bin/activate

# Set your token (replace with your actual token)
export HF_TOKEN=hf_your_token_here

# Download the model (≈ 4.5 GB)
python3 scripts/download_model.py --model solidrust/Mistral-7B-Instruct-v0.3-AWQ
```

The model will be saved to `./models/Mistral-7B-Instruct-v0.3-AWQ/`.

> **Alternative (Llama 3.1 8B AWQ):** If you prefer Llama, first accept the
> license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct,
> then run:
> ```bash
> python3 scripts/download_model.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
> ```

---

## 6. Downloading the Dataset

The ShareGPT dataset contains real user conversations with diverse prompt and
response lengths, providing realistic inference request distributions.

```bash
source .venv/bin/activate
python3 scripts/download_dataset.py
```

This downloads the dataset from HuggingFace and saves a cleaned, filtered
version to `./data/sharegpt_clean.json`. The raw dataset is approximately
180 MB; the cleaned version will be smaller depending on filtering.

You can also manually download the raw dataset from:
> https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

Click **"Files and versions"** → download `ShareGPT_V3_unfiltered_cleaned_split.json`.
Place it in `./data/` and re-run the script with `--local ./data/ShareGPT_V3_unfiltered_cleaned_split.json`.

---

## 7. Running the vLLM Server

Open a **dedicated terminal window** for the server. It must stay running
while experiments execute.

```bash
# Terminal 1 — vLLM Server
source .venv/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
    --model ./models/Mistral-7B-Instruct-v0.3-AWQ \
    --quantization awq \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --host 127.0.0.1 \
    --port 8000 \
    --disable-log-requests
```

Wait until you see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Verify the server is healthy in a second terminal:

```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"healthy"}

curl http://127.0.0.1:8000/v1/models
# Expected: JSON listing Mistral-7B-Instruct-v0.3-AWQ
```

---

## 8. Traffic Capture Setup

Packet capture runs in its own terminal alongside the server and experiments.

```bash
# Terminal 2 — Packet Capture
# Find your loopback interface name first:
ip link show | grep -E "^[0-9]+: lo"
# It will be "lo" on most systems

# Start capture (requires sudo or wireshark group membership)
sudo tcpdump \
    -i lo \
    -w ./results/traces/capture_$(date +%Y%m%d_%H%M%S).pcap \
    -s 0 \
    port 8000
```

Leave this running. Press `Ctrl+C` to stop after experiments finish.
Each experiment run should have its own capture file — stop and restart
tcpdump between experimental conditions.

> **WSL2 Note:** If `lo` does not capture traffic, try the interface name
> `lo0` or use `any`:
> ```bash
> sudo tcpdump -i any -w ./results/traces/capture.pcap port 8000
> ```

---

## 9. Running Experiments

With the vLLM server running (Terminal 1) and tcpdump capturing (Terminal 2),
open a third terminal for experiments.

```bash
# Terminal 3 — Experiments
source .venv/bin/activate
```

### Experiment Matrix

The study tests four concurrency levels × three prompt length categories:

| Concurrency | Prompt Class | Expected Duration |
|---|---|---|
| 1 | short / medium / long | ~5 min each |
| 8 | short / medium / long | ~10 min each |
| 32 | short / medium / long | ~15 min each |
| 64 | short / medium / long | ~20 min each |

Total data collection time: approximately 3–4 hours per full run.

### Running a Single Experiment

```bash
python3 scripts/run_experiments.py \
    --concurrency 1 \
    --prompt-class short \
    --num-requests 100 \
    --output ./results/exp_c1_short.json
```

### Running the Full Experiment Matrix

```bash
python3 scripts/run_experiments.py --full-matrix --output-dir ./results/
```

This runs all 12 conditions sequentially. Stop and restart tcpdump between
conditions if you want per-condition pcap files (recommended). The script
will print a prompt between conditions reminding you to do this.

### What the Script Records

For each request, the script records:

- **Request metadata:** prompt length (tokens), expected response length
- **TTFB (Time to First Byte):** time from request send to first token received,
  used as a proxy for prefill phase duration
- **Inter-token intervals:** timestamps of each streamed token, representing
  decode phase traffic
- **Total duration:** full request completion time
- **Flow size:** approximate bytes sent and received

---

## 10. Analyzing Results

After collecting data, run the analysis script to produce all plots and
summary statistics.

```bash
source .venv/bin/activate

python3 scripts/analyze.py \
    --results-dir ./results/ \
    --traces-dir ./results/traces/ \
    --output-dir ./results/plots/
```

This produces:

- `flow_size_cdf.png` — CDF of flow sizes for prefill vs. decode phases
- `ttfb_by_concurrency.png` — TTFB distributions across concurrency levels
- `inter_token_intervals.png` — decode phase timing distributions
- `burst_analysis.png` — burst magnitude and duration characterization
- `arrival_process.png` — inter-request arrival time analysis
- `summary_stats.csv` — full numerical summary table

Open plots from the `./results/plots/` directory with any image viewer.

---

## 11. Directory Structure

```
llm-inference-traffic/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── models/                     # Downloaded model weights (git-ignored)
├── data/
│   ├── sharegpt_clean.json     # Processed ShareGPT dataset
│   └── .gitkeep
├── results/
│   ├── traces/                 # Raw pcap files from tcpdump
│   ├── plots/                  # Output figures
│   ├── exp_c1_short.json       # Per-experiment result files
│   └── ...
└── scripts/
    ├── download_model.py       # HuggingFace model downloader
    ├── download_dataset.py     # ShareGPT dataset downloader/cleaner
    ├── run_experiments.py      # Main experiment runner
    └── analyze.py              # Analysis and plotting
```

---

## 12. Troubleshooting

**vLLM install fails with CUDA error:**
Make sure your CUDA version matches the vLLM wheel. Check:
```bash
nvcc --version      # CUDA version
python3 -c "import torch; print(torch.version.cuda)"   # PyTorch CUDA
```
If they mismatch, reinstall PyTorch first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Out of VRAM when starting vLLM:**
Reduce `--gpu-memory-utilization` to 0.75, or lower `--max-model-len` to 2048.
Close other GPU-using applications (games, browsers with hardware acceleration).

**tcpdump: Permission denied:**
Either use `sudo`, or confirm you added yourself to the wireshark group and
ran `newgrp wireshark` (or logged out and back in).

**vLLM server returns 422 errors:**
The model name in your request must exactly match the name shown by
`/v1/models`. Update the `--model-name` argument in `run_experiments.py`
to match.

**WSL2: nvidia-smi not found:**
Install the NVIDIA driver on the Windows host (not inside WSL2). WSL2
inherits the Windows host GPU driver. Download from:
https://www.nvidia.com/Download/index.aspx

**ShareGPT download fails:**
Download the file manually from HuggingFace (see Section 6) and pass
`--local ./data/ShareGPT_V3_unfiltered_cleaned_split.json` to the download script.

---

## Quick Reference: Terminal Layout

| Terminal | Purpose | Command |
|---|---|---|
| Terminal 1 | vLLM server | `python3 -m vllm.entrypoints.openai.api_server ...` |
| Terminal 2 | tcpdump capture | `sudo tcpdump -i lo -w capture.pcap port 8000` |
| Terminal 3 | Experiments | `python3 scripts/run_experiments.py ...` |
| Terminal 4 | Monitoring (optional) | `watch -n1 nvidia-smi` |

---

*For questions about methodology, refer to the research proposal (`research_proposal.pdf`).*
