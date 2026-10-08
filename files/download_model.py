#!/usr/bin/env python3
"""
download_model.py
Downloads a quantized LLM from HuggingFace Hub into ./models/.

Usage:
    python3 scripts/download_model.py
    python3 scripts/download_model.py --model solidrust/Mistral-7B-Instruct-v0.3-AWQ
    python3 scripts/download_model.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login
from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_MODEL = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
MODELS_DIR = Path(__file__).parent.parent / "models"


def get_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        console.print(
            "[yellow]HF_TOKEN environment variable not set.[/yellow]\n"
            "Set it with:\n"
            "  export HF_TOKEN=hf_your_token_here\n\n"
            "Get your token at: [link]https://huggingface.co/settings/tokens[/link]"
        )
        token = input("Or paste your HuggingFace token now (will not be saved): ").strip()
        if not token:
            console.print("[red]No token provided. Exiting.[/red]")
            sys.exit(1)
    return token


def download_model(model_id: str, token: str) -> Path:
    model_name = model_id.split("/")[-1]
    local_dir = MODELS_DIR / model_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if local_dir.exists() and any(local_dir.iterdir()):
        console.print(
            f"[green]Model already exists at {local_dir}.[/green]\n"
            "Delete the directory and re-run to force re-download."
        )
        return local_dir

    console.print(Panel(
        f"[bold]Downloading:[/bold] {model_id}\n"
        f"[bold]Destination:[/bold] {local_dir}\n\n"
        "This may take 10–20 minutes depending on your connection.\n"
        "The model is approximately 4–5 GB.",
        title="Model Download",
        border_style="blue"
    ))

    login(token=token, add_to_git_credential=False)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )

    console.print(f"\n[green]Model saved to: {local_dir}[/green]")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download a quantized LLM from HuggingFace.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold blue]LLM Inference Traffic Research[/bold blue]\n"
        "Model Downloader",
        border_style="blue"
    ))

    token = get_token()
    local_dir = download_model(args.model, token)

    console.print(
        "\n[bold green]Done![/bold green]\n\n"
        "Start the vLLM server with:\n\n"
        f"  python3 -m vllm.entrypoints.openai.api_server \\\n"
        f"    --model {local_dir} \\\n"
        f"    --quantization awq \\\n"
        f"    --max-model-len 4096 \\\n"
        f"    --host 127.0.0.1 \\\n"
        f"    --port 8000"
    )


if __name__ == "__main__":
    main()
