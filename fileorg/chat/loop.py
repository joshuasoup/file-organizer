from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import typer

from fileorg.config import load_config
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
    text_embedder = None

    def get_text_embedder():
        nonlocal text_embedder
        if text_embedder is None:
            # Lazy load to avoid startup lag; first call may take a few seconds.
            console.print(
                "[yellow]Loading text embedding model (first time may take a few seconds)...[/yellow]"
            )
            from fileorg.embeddings import build_text_embedder

            text_embedder = build_text_embedder(config)
        return text_embedder

    system_prompt = (
        "You are FileOrg, a local file organization assistant. "
        "Use tools to search files, find duplicates, find stale files, suggest folder structures, "
        "and preview move plans. Never delete files. Default to dry-run/preview."
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
            user_input = console.input("[bold green]you › [/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("Goodbye.")
            break
        messages.append({"role": "user", "content": user_input})

        with console.status("[cyan]fileorg is thinking…[/cyan]"):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tool_definitions(),
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
                        "text_embedder": get_text_embedder(),
                        "client": client,
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
            with console.status("[cyan]fileorg is thinking…[/cyan]"):
                response = client.chat.completions.create(
                    model="gpt-4o", messages=messages, tools=tool_definitions()
                )
            message = response.choices[0].message

        if message.content:
            console.print(Panel(Markdown(message.content), title="fileorg", border_style="cyan"))
            messages.append({"role": "assistant", "content": message.content})
