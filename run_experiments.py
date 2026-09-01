#!/usr/bin/env python3
"""
run_experiments.py
Drives vLLM with ShareGPT prompts at varying concurrency levels and records
detailed per-request timing: TTFB, inter-token intervals, flow sizes, and
total duration. These timing measurements serve as the primary data source
for characterizing prefill and decode phase network traffic.

Usage:
    # Single condition
    python3 scripts/run_experiments.py --concurrency 1 --prompt-class short --num-requests 100

    # Full experiment matrix (all concurrency x prompt-class combinations)
    python3 scripts/run_experiments.py --full-matrix --output-dir ./results/

    # Dry run (validate setup without sending real requests)
    python3 scripts/run_experiments.py --dry-run
"""

import asyncio
import json
import sys
import time
import argparse
import random
import signal
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASET_FILE = DATA_DIR / "sharegpt_clean.json"

VLLM_BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = None  # auto-detected from /v1/models

CONCURRENCY_LEVELS = [1, 8, 32, 64]
PROMPT_CLASSES = ["short", "medium", "long"]
DEFAULT_NUM_REQUESTS = 150  # per condition


@dataclass
class RequestResult:
    request_id: int
    prompt_class: str
    prompt_chars: int
    response_chars_expected: int
    concurrency: int
    send_time: float             # Unix timestamp when request was sent
    ttfb: float                  # seconds from send to first token (prefill proxy)
    total_duration: float        # seconds from send to last token
    tokens_received: int         # number of tokens in streamed response
    inter_token_intervals: list  # list of floats (seconds between successive tokens)
    request_bytes_sent: int      # approximate bytes in HTTP request body
    response_bytes_received: int # approximate bytes in HTTP response body
    success: bool
    error: str = ""


async def get_model_name(client: httpx.AsyncClient) -> str:
    try:
        resp = await client.get("/v1/models", timeout=10.0)
        data = resp.json()
        name = data["data"][0]["id"]
        return name
    except Exception as e:
        console.print(f"[red]Could not reach vLLM server at {VLLM_BASE_URL}.[/red]")
        console.print(f"  Error: {e}")
        console.print(
            "\nMake sure the server is running:\n"
            "  python3 -m vllm.entrypoints.openai.api_server \\\n"
            "    --model ./models/Mistral-7B-Instruct-v0.3-AWQ \\\n"
            "    --quantization awq --host 127.0.0.1 --port 8000"
        )
        sys.exit(1)


async def send_request(
    client: httpx.AsyncClient,
    model_name: str,
    request_id: int,
    prompt: str,
    prompt_class: str,
    prompt_chars: int,
    response_chars_expected: int,
    concurrency: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:

    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True,
    }
    payload_bytes = len(json.dumps(payload).encode("utf-8"))

    async with semaphore:
        send_time = time.perf_counter()
        unix_send = time.time()
        first_token_time = None
        inter_token_intervals = []
        last_token_time = None
        total_response_bytes = 0
        tokens_received = 0
        full_response = ""
        error_msg = ""

        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=120.0,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        now = time.perf_counter()
                        total_response_bytes += len(content.encode("utf-8"))
                        full_response += content
                        tokens_received += 1

                        if first_token_time is None:
                            first_token_time = now
                        else:
                            inter_token_intervals.append(now - last_token_time)

                        last_token_time = now

            success = True

        except Exception as e:
            error_msg = str(e)
            success = False

        end_time = time.perf_counter()
        ttfb = (first_token_time - send_time) if first_token_time else -1.0
        total_duration = end_time - send_time

        return RequestResult(
            request_id=request_id,
            prompt_class=prompt_class,
            prompt_chars=prompt_chars,
            response_chars_expected=response_chars_expected,
            concurrency=concurrency,
            send_time=unix_send,
            ttfb=ttfb,
            total_duration=total_duration,
            tokens_received=tokens_received,
            inter_token_intervals=inter_token_intervals,
            request_bytes_sent=payload_bytes,
            response_bytes_received=total_response_bytes,
            success=success,
            error=error_msg,
        )


def load_prompts(prompt_class: str, num_requests: int) -> list[dict]:
    if not DATASET_FILE.exists():
        console.print(f"[red]Dataset not found at {DATASET_FILE}.[/red]")
        console.print("Run: python3 scripts/download_dataset.py")
        sys.exit(1)

    with open(DATASET_FILE) as f:
        data = json.load(f)

    pool = [s for s in data if s["length_class"] == prompt_class]

    if len(pool) < num_requests:
        console.print(
            f"[yellow]Warning: only {len(pool)} samples available for class '{prompt_class}', "
            f"requested {num_requests}. Using all available with repetition.[/yellow]"
        )
        pool = pool * (num_requests // len(pool) + 1)

    random.shuffle(pool)
    return pool[:num_requests]


async def run_condition(
    model_name: str,
    concurrency: int,
    prompt_class: str,
    num_requests: int,
    output_file: Path,
) -> list[RequestResult]:

    prompts = load_prompts(prompt_class, num_requests)
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    console.print(
        f"\n[bold blue]Running condition:[/bold blue] "
        f"concurrency={concurrency}, class={prompt_class}, n={num_requests}"
    )

    async with httpx.AsyncClient(base_url=VLLM_BASE_URL) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Sending requests...", total=num_requests)

            async def wrapped(idx: int, sample: dict):
                result = await send_request(
                    client=client,
                    model_name=model_name,
                    request_id=idx,
                    prompt=sample["prompt"],
                    prompt_class=sample["length_class"],
                    prompt_chars=sample["prompt_chars"],
                    response_chars_expected=sample["response_chars"],
                    concurrency=concurrency,
                    semaphore=semaphore,
                )
                results.append(result)
                progress.advance(task)

            await asyncio.gather(*[
                wrapped(i, sample)
                for i, sample in enumerate(prompts)
            ])

    success_count = sum(1 for r in results if r.success)
    console.print(
        f"  [green]Completed:[/green] {success_count}/{num_requests} successful"
    )
    if success_count < num_requests:
        fail_sample = next(r for r in results if not r.success)
        console.print(f"  [yellow]Sample error: {fail_sample.error}[/yellow]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    console.print(f"  Saved to: {output_file}")

    return results


def print_condition_summary(results: list):
    successful = [r for r in results if r.success and r.ttfb > 0]
    if not successful:
        console.print("[red]No successful results to summarize.[/red]")
        return

    import statistics
    ttfbs = [r.ttfb for r in successful]
    durations = [r.total_duration for r in successful]
    token_counts = [r.tokens_received for r in successful]
    all_iti = [iti for r in successful for iti in r.inter_token_intervals]

    table = Table(title="Condition Summary", border_style="blue")
    table.add_column("Metric")
    table.add_column("Median")
    table.add_column("P95")
    table.add_column("Mean")

    def row(name, values):
        s = sorted(values)
        p95 = s[int(len(s) * 0.95)]
        return [name, f"{statistics.median(s):.3f}", f"{p95:.3f}", f"{statistics.mean(s):.3f}"]

    table.add_row(*row("TTFB (s)", ttfbs))
    table.add_row(*row("Total duration (s)", durations))
    table.add_row(*row("Tokens received", token_counts))
    if all_iti:
        table.add_row(*row("Inter-token interval (s)", all_iti))

    console.print(table)


async def main_async(args):
    async with httpx.AsyncClient(base_url=VLLM_BASE_URL) as client:
        model_name = await get_model_name(client)

    console.print(f"[green]Connected to vLLM. Model: {model_name}[/green]")

    if args.dry_run:
        console.print("[yellow]Dry run: server is reachable. Exiting without sending requests.[/yellow]")
        return

    if args.full_matrix:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = Path(args.output_dir)
        total_conditions = len(CONCURRENCY_LEVELS) * len(PROMPT_CLASSES)

        console.print(Panel(
            f"[bold]Full experiment matrix:[/bold] "
            f"{len(CONCURRENCY_LEVELS)} concurrency levels × {len(PROMPT_CLASSES)} prompt classes "
            f"= {total_conditions} conditions\n"
            f"{args.num_requests} requests per condition\n\n"
            "[yellow]Between each condition, stop and restart tcpdump to keep traces separate.[/yellow]\n"
            "The script will pause and prompt you between conditions.",
            border_style="blue"
        ))

        for i, concurrency in enumerate(CONCURRENCY_LEVELS):
            for j, prompt_class in enumerate(PROMPT_CLASSES):
                condition_num = i * len(PROMPT_CLASSES) + j + 1
                fname = f"exp_c{concurrency}_{prompt_class}.json"
                output_file = output_dir / fname

                if output_file.exists():
                    console.print(f"[yellow]Skipping {fname} — already exists.[/yellow]")
                    continue

                console.print(
                    f"\n[bold]===== Condition {condition_num}/{total_conditions} =====[/bold]"
                )
                console.print(
                    f"[yellow]ACTION REQUIRED:[/yellow] "
                    f"In Terminal 2, stop any running tcpdump (Ctrl+C) then restart:\n\n"
                    f"  sudo tcpdump -i lo -w ./results/traces/trace_c{concurrency}_{prompt_class}.pcap port 8000\n"
                )
                input("Press Enter when tcpdump is capturing...")

                results = await run_condition(
                    model_name=model_name,
                    concurrency=concurrency,
                    prompt_class=prompt_class,
                    num_requests=args.num_requests,
                    output_file=output_file,
                )
                print_condition_summary(results)

        console.print("\n[bold green]All conditions complete![/bold green]")
        console.print("Stop tcpdump (Ctrl+C in Terminal 2), then run:\n  python3 scripts/analyze.py")

    else:
        # Single condition
        output_file = Path(args.output) if args.output else \
            RESULTS_DIR / f"exp_c{args.concurrency}_{args.prompt_class}.json"

        results = await run_condition(
            model_name=model_name,
            concurrency=args.concurrency,
            prompt_class=args.prompt_class,
            num_requests=args.num_requests,
            output_file=output_file,
        )
        print_condition_summary(results)


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference traffic experiments.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests (single condition).")
    parser.add_argument("--prompt-class", choices=["short", "medium", "long"], default="short", help="Prompt length class (single condition).")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS, help=f"Requests per condition (default: {DEFAULT_NUM_REQUESTS}).")
    parser.add_argument("--output", default=None, help="Output JSON file (single condition).")
    parser.add_argument("--full-matrix", action="store_true", help="Run all concurrency × prompt-class combinations.")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="Output directory for full matrix mode.")
    parser.add_argument("--dry-run", action="store_true", help="Check server connectivity only, do not send requests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt shuffling.")
    args = parser.parse_args()

    random.seed(args.seed)

    console.print(Panel(
        "[bold blue]LLM Inference Traffic Research[/bold blue]\n"
        "Experiment Runner",
        border_style="blue"
    ))

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
