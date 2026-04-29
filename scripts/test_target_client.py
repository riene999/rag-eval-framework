from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_eval.clients import HttpTargetAgentClient, MockTargetAgentClient

try:
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test a target RAG/Agent client.")
    parser.add_argument("--mock", action="store_true", help="Use the built-in mock target client.")
    parser.add_argument("--base-url", help="Base URL for the external target service, e.g. http://localhost:8000.")
    parser.add_argument("--endpoint", default="/ask", help="Target ask endpoint. Defaults to /ask.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--question", required=True, help="Question to send to the target.")
    parser.add_argument("--top-k", type=int, default=5, help="Requested retrieval top_k.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mock:
        client = MockTargetAgentClient()
    elif args.base_url:
        client = HttpTargetAgentClient(args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    else:
        print_error("Error: provide either --mock or --base-url.")
        return 2

    try:
        response = client.ask(args.question, top_k=args.top_k)
    except Exception as exc:
        print_error(f"Target client call failed: {exc}")
        return 1

    print_response(response)
    return 0


def print_error(message: str) -> None:
    """Print an error message with Rich when available."""
    if RICH_AVAILABLE and console is not None:
        console.print(f"[bold red]{message}[/bold red]")
    else:
        print(message, file=sys.stderr)


def print_response(response: dict[str, Any]) -> None:
    """Pretty-print a target response."""
    if RICH_AVAILABLE and console is not None:
        answer = str(response.get("answer", ""))
        retrieved_count = len(response.get("retrieved_chunks", []))
        latency = response.get("latency_ms")
        client_latency = response.get("client_latency_ms")
        console.print(
            Panel(
                f"[bold]Retrieved chunks:[/bold] {retrieved_count}\n"
                f"[bold]Target latency:[/bold] {latency} ms\n"
                f"[bold]Client latency:[/bold] {client_latency} ms\n"
                f"[bold]Answer preview:[/bold] {answer[:240]}",
                title="Target Client Response",
                border_style="green",
            )
        )
        console.print(JSON(json.dumps(response, ensure_ascii=False)))
    else:
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
