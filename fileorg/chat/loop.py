from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
import typer

from fileorg.config import load_config
from fileorg.embeddings import build_text_embedder
from fileorg.chat.tools import (
    ToolResult,
    tool_definitions,
    tool_duplicates,
    tool_preview_moves,
    tool_search,
    tool_stale,
    tool_suggest_structure,
)
from fileorg.store import ChromaStore, MetadataStore

console = Console()
_STATUS_UPDATE_INTERVAL = 0.2


def _run_with_timer(label: str, fn: Callable[[], Any]) -> tuple[Any, float]:
    """Show a live timer while running a blocking call."""
    start = time.perf_counter()
    stop_event = threading.Event()
    status = console.status("")

    def updater() -> None:
        while not stop_event.wait(_STATUS_UPDATE_INTERVAL):
            elapsed = time.perf_counter() - start
            status.update(f"[cyan]{label} {int(elapsed)}s[/cyan]")

    with status:
        status.update(f"[cyan]{label} 0s[/cyan]")
        thread = threading.Thread(target=updater, daemon=True)
        thread.start()
        try:
            result = fn()
        finally:
            stop_event.set()
            thread.join()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _handle_tool_call(name: str, args: dict[str, Any], deps: dict) -> ToolResult:
    if name == "search_files":
        return tool_search(
            query=args.get("query", ""),
            limit=int(args.get("limit", 5)),
            chroma=deps["chroma"],
            embedder=deps["text_embedder"],
        )
    if name == "find_duplicates":
        return tool_duplicates(metadata=deps["metadata"])
    if name == "find_stale_files":
        return tool_stale(days=int(args.get("days", 180)), metadata=deps["metadata"])
    if name == "suggest_structure":
        return tool_suggest_structure(
            chroma=deps["chroma"],
            client=deps["client"],
            config=deps["config"],
            min_cluster_size=int(args.get("min_cluster_size", 3)),
            min_samples=int(args.get("min_samples", 2)),
        )
    if name == "preview_moves":
        plan = args.get("plan", [])
        return tool_preview_moves(plan=plan)
    raise ValueError(f"Unknown tool: {name}")


def chat_loop() -> None:
    config = load_config(create=True)
    client = OpenAI()
    chroma = ChromaStore(config.chroma_dir())
    metadata = MetadataStore(config.metadata_db_path())
    text_embedder, _ = _run_with_timer(
        "Loading text embedder…", lambda: build_text_embedder(config)
    )

    def limited_messages(
        max_chars: int = 12000, max_messages: int = 18
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system = messages[0]
        keep_indices: list[int] = []
        total_chars = len(str(system.get("content", "")))
        idx = len(messages) - 1

        def _tool_call_id(tc: Any) -> str | None:
            if isinstance(tc, dict):
                return tc.get("id") or tc.get("tool_call_id")
            return getattr(tc, "id", None)

        while idx >= 1:
            msg = messages[idx]
            role = msg.get("role")
            msg_len = len(str(msg.get("content", "")))

            if role == "tool":
                tool_id = msg.get("tool_call_id")
                parent_idx = None
                for j in range(idx - 1, 0, -1):
                    prev = messages[j]
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        if any(_tool_call_id(tc) == tool_id for tc in prev.get("tool_calls")):
                            parent_idx = j
                            break
                if parent_idx is None:
                    idx -= 1
                    continue

                pair_len = msg_len
                if parent_idx not in keep_indices:
                    pair_len += len(str(messages[parent_idx].get("content", "")))
                if (
                    len(keep_indices) + (2 if parent_idx not in keep_indices else 1)
                    > max_messages
                    or total_chars + pair_len > max_chars
                ):
                    break

                if parent_idx not in keep_indices:
                    keep_indices.append(parent_idx)
                    total_chars += len(str(messages[parent_idx].get("content", "")))
                keep_indices.append(idx)
                total_chars += msg_len
                idx = parent_idx - 1
                continue

            if len(keep_indices) + 1 > max_messages or total_chars + msg_len > max_chars:
                break
            keep_indices.append(idx)
            total_chars += msg_len
            idx -= 1

        keep_indices_sorted = sorted(set(keep_indices))
        return [system] + [messages[i] for i in keep_indices_sorted]

    system_prompt = (
        "You are FileOrg, a local file organization assistant. "
        "Use tools to search files, find duplicates, find stale files, and organize files. "
        "For any request to organize/restructure/move files, always return a preview move plan and tree (dry-run only), "
        "using suggest_structure or preview_moves as appropriate. Never delete or actually move files."
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    console.print(
        Panel(
            "[bold cyan]FileOrg[/bold cyan] chat (GPT-4o)\n"
            "[dim]Type 'exit' to quit. Tools: search, duplicates, stale, structure, preview.[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("Goodbye.")
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})

        response, _ = _run_with_timer(
            "Thinking…",
            lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=limited_messages(),
                tools=tool_definitions(),
            ),
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if tool_calls:
            # Keep the assistant message with tool_calls in history for proper threading.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )
            for call in tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = _handle_tool_call(
                    call.function.name,
                    args,
                    deps={
                        "chroma": chroma,
                        "metadata": metadata,
                        "text_embedder": text_embedder,
                        "client": client,
                        "config": config,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": result.name,
                        "content": result.content,
                    }
                )
            response, _ = _run_with_timer(
                "Thinking…",
                lambda: client.chat.completions.create(
                    model="gpt-4o", messages=limited_messages(), tools=tool_definitions()
                ),
            )
            message = response.choices[0].message

        if message.content:
            console.print(f"[pink]  •[/pink] [bright_black]{message.content}[/]")
            messages.append({"role": "assistant", "content": message.content})
