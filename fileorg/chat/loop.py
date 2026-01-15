from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
import typer

from fileorg.config import load_config
from fileorg.embeddings import build_text_embedder, build_image_embedder
from fileorg.undo import clear_last_action, save_last_move
from fileorg.chat.tools import (
    ToolResult,
    tool_definitions,
    tool_duplicates,
    tool_delete,
    tool_move,
    tool_preview_moves,
    tool_search,
    tool_suggest_structure,
    tool_undo_last_action,
)
from fileorg.chat.tools.plan_ops import build_move_plan, parse_structure_suggestions
from fileorg.chat.tools.previews import (
    display_delete_preview,
    display_duplicates,
    display_move_preview,
    display_search_results,
    display_structure_tree,
)
from fileorg.chat.system_prompt import get_system_prompt
from fileorg.indexer import Indexer
from fileorg.store import ChromaStore, MetadataStore

console = Console()
_STATUS_UPDATE_INTERVAL = 0.2
AUTO_STRUCTURE_TREE = os.environ.get("FILEORG_AUTO_STRUCTURE_TREE", "0") == "1"


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


def _refresh_index(config, metadata, chroma, text_embedder) -> None:
    """Reindex after moves/deletes to keep stores in sync."""
    try:
        image_embedder = build_image_embedder(config)
        indexer = Indexer(
            config=config,
            metadata=metadata,
            chroma=chroma,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
        )
        _run_with_timer("Re-indexing after changes…", lambda: indexer.run(full=True))
    except Exception as exc:
        console.print(f"[yellow]Index refresh failed: {exc}[/yellow]")


def _auto_index_on_start(config, metadata, chroma, text_embedder) -> None:
    """Quick incremental index when chat starts to pick up new files."""
    try:
        image_embedder = build_image_embedder(config)
        indexer = Indexer(
            config=config,
            metadata=metadata,
            chroma=chroma,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
        )
    except Exception as exc:
        console.print(f"[yellow]Skipping auto-index (missing embedder): {exc}[/yellow]")
        return

    try:
        stats, elapsed = _run_with_timer(
            "Auto-indexing new files…", lambda: indexer.run(full=False)
        )
    except Exception as exc:
        console.print(f"[yellow]Auto-index failed: {exc}[/yellow]")
        return

    if stats.indexed or stats.removed:
        console.print(
            f"[green]Auto-index complete[/green] "
            f"(indexed {stats.indexed}, skipped {stats.skipped}, removed {stats.removed}) "
            f"in {elapsed:.1f}s."
        )
    else:
        console.print(f"[dim]Index already up to date ({elapsed:.1f}s).[/dim]")





def _handle_tool_call(name: str, args: dict[str, Any], deps: dict) -> ToolResult:
    # usage_callback is passed in deps if available
    if name == "search_files":
        return tool_search(
            query=args.get("query", ""),
            limit=int(args.get("limit", 5)),
            chroma=deps["chroma"],
            embedder=deps["text_embedder"],
            metadata=deps.get("metadata"),
        )
    if name == "find_duplicates":
        return tool_duplicates(metadata=deps["metadata"])
    if name == "suggest_structure":
        # Wrap in timer to show progress for this potentially slow operation
        def _run_structure_tool():
            config = deps.get("config")
            result = tool_suggest_structure(
                chroma=deps["chroma"],
                client=deps["client"],
                min_cluster_size=int(args.get("min_cluster_size", 3)),
                min_samples=int(args.get("min_samples", 2)),
                scan_root=config.scan.root if config is not None else None,
                config=config,
                usage_callback=deps.get("usage_callback"),
            )
            return result
        
        result, _ = _run_with_timer("Analyzing file structure…", _run_structure_tool)
        return result
    if name == "preview_moves":
        plan = args.get("plan", [])
        return tool_preview_moves(plan=plan)
    if name == "move_files":
        plan = args.get("plan", [])
        return tool_move(plan=plan)
    if name == "delete_items":
        items = args.get("items", [])
        return tool_delete(items=items)
    if name == "undo_last_action":
        return tool_undo_last_action()
    raise ValueError(f"Unknown tool: {name}")


def chat_loop() -> None:
    config = load_config(create=True)
    client = OpenAI(timeout=60.0)  # 60 second timeout for all API calls
    chroma = ChromaStore(config.chroma_dir())
    metadata = MetadataStore(config.metadata_db_path())
    text_embedder, _ = _run_with_timer(
        "Loading text embedder…", lambda: build_text_embedder(config)
    )
    _auto_index_on_start(config, metadata, chroma, text_embedder)

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
        if not keep_indices_sorted:
            return [system]

        # Ensure any assistant tool_calls have matching tool messages (and vice versa)
        assistant_ids: dict[int, list[str]] = {}
        tool_ids: dict[str, list[int]] = {}
        for idx in keep_indices_sorted:
            msg = messages[idx]
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                ids = [
                    _tool_call_id(tc) for tc in msg.get("tool_calls") if _tool_call_id(tc)
                ]
                if ids:
                    assistant_ids[idx] = ids
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    tool_ids.setdefault(tid, []).append(idx)

        valid_tool_ids = {
            tid for ids in assistant_ids.values() for tid in ids if tid in tool_ids
        }
        filtered_indices: list[int] = []
        for idx in keep_indices_sorted:
            msg = messages[idx]
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                ids = [
                    _tool_call_id(tc) for tc in msg.get("tool_calls") if _tool_call_id(tc)
                ]
                if not any(tid in valid_tool_ids for tid in ids):
                    continue
            if role == "tool":
                tid = msg.get("tool_call_id")
                if tid not in valid_tool_ids:
                    continue
            filtered_indices.append(idx)

        return [system] + [messages[i] for i in filtered_indices]

    system_prompt = get_system_prompt()

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    console.print(
        Panel(
            "[bold cyan]FileOrg[/bold cyan] chat (GPT-4o)\n"
            "[dim]Type 'exit' to quit. Tools: search, duplicates, structure, preview, move, delete, undo.[/dim]",
            border_style="cyan",
        )
    )

    # Track token usage throughout the session
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    def _accumulate_usage(response) -> None:
        """Accumulate token usage from an API response."""
        nonlocal total_prompt_tokens, total_completion_tokens, total_tokens
        if hasattr(response, "usage") and response.usage:
            total_prompt_tokens += response.usage.prompt_tokens or 0
            total_completion_tokens += response.usage.completion_tokens or 0
            total_tokens += response.usage.total_tokens or 0

    try:
        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.strip().lower() in {"exit", "quit"}:
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
            _accumulate_usage(response)
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
                action_applied = False
                action_messages: list[str] = []
                for call in tool_calls:
                    console.print(f"[dim]Calling tool: {call.function.name}[/dim]")
                    args = json.loads(call.function.arguments or "{}")
                    try:
                        deps = {
                            "chroma": chroma,
                            "metadata": metadata,
                            "text_embedder": text_embedder,
                            "client": client,
                            "config": config,
                            "usage_callback": _accumulate_usage,
                        }
                        result = _handle_tool_call(call.function.name, args, deps=deps)
                        if call.function.name == "search_files":
                            # Display search results in a clean, user-friendly format
                            display_search_results(
                                result.content,
                                console=console,
                                query=args.get("query"),
                            )
                            # Suppress verbose debug output for search
                            console.print(f"[dim]Tool {call.function.name} completed[/dim]")
                        elif call.function.name == "find_duplicates":
                            # Display duplicates in a clean, user-friendly format
                            display_duplicates(
                                result.content,
                                console=console,
                            )
                            # Suppress verbose debug output for duplicates
                            console.print(f"[dim]Tool {call.function.name} completed[/dim]")
                        elif call.function.name == "suggest_structure":
                            scan_root = deps["config"].scan.root if deps.get("config") is not None else None
                            if AUTO_STRUCTURE_TREE:
                                display_structure_tree(
                                    result.content,
                                    console=console,
                                    scan_root=scan_root,
                                    after_apply=lambda: _refresh_index(config, metadata, chroma, text_embedder),
                                )
                            suggestions = parse_structure_suggestions(result.content)
                            if suggestions:
                                plan = build_move_plan(suggestions, scan_root=scan_root)
                                preview_result = display_move_preview(
                                    json.dumps(plan),
                                    console=console,
                                    scan_root=scan_root,
                                    after_apply=lambda: _refresh_index(config, metadata, chroma, text_embedder),
                                )
                                if preview_result and preview_result.get("action") == "approve":
                                    stats = preview_result.get("stats") or {}
                                    applied_moves = stats.get("applied") or []
                                    if applied_moves and scan_root is not None:
                                        save_last_move(applied_moves, scan_root)
                                    else:
                                        clear_last_action()
                                    summary = (
                                        f"Move plan approved and applied "
                                        f"(moved {stats.get('moved', 0)}, "
                                        f"missing {stats.get('skipped_missing', 0)}, "
                                        f"conflicts {stats.get('skipped_conflict', 0)}, "
                                        f"outside-root {stats.get('skipped_outside', 0)}, "
                                        f"errors {stats.get('errors', 0)})."
                                    )
                                    action_messages.append(summary)
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": summary,
                                        }
                                    )
                                    action_applied = True
                        elif call.function.name == "preview_moves":
                            preview_result = display_move_preview(
                                result.content,
                                console=console,
                                scan_root=deps["config"].scan.root if deps.get("config") is not None else None,
                                after_apply=lambda: _refresh_index(config, metadata, chroma, text_embedder),
                            )
                            if preview_result and preview_result.get("action") == "approve":
                                stats = preview_result.get("stats") or {}
                                applied_moves = stats.get("applied") or []
                                scan_root = deps["config"].scan.root if deps.get("config") is not None else None
                                if applied_moves and scan_root is not None:
                                    save_last_move(applied_moves, scan_root)
                                else:
                                    clear_last_action()
                                summary = (
                                    f"Move plan approved and applied "
                                    f"(moved {stats.get('moved', 0)}, "
                                    f"missing {stats.get('skipped_missing', 0)}, "
                                    f"conflicts {stats.get('skipped_conflict', 0)}, "
                                    f"outside-root {stats.get('skipped_outside', 0)}, "
                                    f"errors {stats.get('errors', 0)})."
                                )
                                action_messages.append(summary)
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": summary,
                                    }
                                )
                                action_applied = True
                        elif call.function.name == "move_files":
                            preview_result = display_move_preview(
                                result.content,
                                console=console,
                                scan_root=deps["config"].scan.root if deps.get("config") is not None else None,
                                after_apply=lambda: _refresh_index(config, metadata, chroma, text_embedder),
                            )
                            if preview_result and preview_result.get("action") == "approve":
                                stats = preview_result.get("stats") or {}
                                applied_moves = stats.get("applied") or []
                                scan_root = deps["config"].scan.root if deps.get("config") is not None else None
                                if applied_moves and scan_root is not None:
                                    save_last_move(applied_moves, scan_root)
                                else:
                                    clear_last_action()
                                summary = (
                                    f"Move plan approved and applied "
                                    f"(moved {stats.get('moved', 0)}, "
                                    f"missing {stats.get('skipped_missing', 0)}, "
                                    f"conflicts {stats.get('skipped_conflict', 0)}, "
                                    f"outside-root {stats.get('skipped_outside', 0)}, "
                                    f"errors {stats.get('errors', 0)})."
                                )
                                action_messages.append(summary)
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": summary,
                                    }
                                )
                                action_applied = True
                        elif call.function.name == "delete_items":
                            preview_result = display_delete_preview(
                                result.content,
                                console=console,
                                scan_root=deps["config"].scan.root if deps.get("config") is not None else None,
                                after_apply=lambda: _refresh_index(config, metadata, chroma, text_embedder),
                            )
                            if preview_result and preview_result.get("action") == "approve":
                                stats = preview_result.get("stats") or {}
                                summary = (
                                    f"Delete plan approved and applied "
                                    f"(deleted {stats.get('deleted', 0)}, "
                                    f"missing {stats.get('skipped_missing', 0)}, "
                                    f"outside-root {stats.get('skipped_outside', 0)}, "
                                    f"errors {stats.get('errors', 0)})."
                                )
                                action_messages.append(summary)
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": summary,
                                    }
                                )
                                action_applied = True
                        elif call.function.name == "undo_last_action":
                            undo_payload = {}
                            try:
                                undo_payload = json.loads(result.content or "{}")
                            except Exception:
                                undo_payload = {}
                            error = undo_payload.get("error")
                            if error:
                                console.print(f"[yellow]{error}[/yellow]")
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": error,
                                    }
                                )
                            else:
                                plan = undo_payload.get("plan") or []
                                scan_root = deps["config"].scan.root if deps.get("config") is not None else None
                                preview_result = display_move_preview(
                                    json.dumps(plan),
                                    console=console,
                                    scan_root=scan_root,
                                    after_apply=lambda: _refresh_index(config, metadata, chroma, text_embedder),
                                )
                                if preview_result and preview_result.get("action") == "approve":
                                    stats = preview_result.get("stats") or {}
                                    clear_last_action()
                                    summary = (
                                        f"Undo applied "
                                        f"(moved {stats.get('moved', 0)}, "
                                        f"missing {stats.get('skipped_missing', 0)}, "
                                        f"conflicts {stats.get('skipped_conflict', 0)}, "
                                        f"outside-root {stats.get('skipped_outside', 0)}, "
                                        f"errors {stats.get('errors', 0)})."
                                    )
                                    action_messages.append(summary)
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": summary,
                                        }
                                    )
                                    action_applied = True
                        else:
                            # Default: show debug output for tools without special handling
                            console.print(f"[dim]Tool {call.function.name} completed[/dim]")
                            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
                            console.print(f"[dim]Result preview: {content_preview}[/dim]")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": result.name,
                                "content": result.content,
                            }
                        )
                    except Exception as e:
                        # If tool call fails, add error message to conversation
                        error_msg = f"Error calling {call.function.name}: {str(e)}"
                        console.print(f"[red]Error: {error_msg}[/red]")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": call.function.name,
                                "content": error_msg,
                            }
                        )
                if action_applied:
                    for text in action_messages:
                        console.print(f"[pink]  •[/pink] [white]{text}[/]")
                    continue
                else:
                    response, _ = _run_with_timer(
                        "Thinking…",
                        lambda: client.chat.completions.create(
                            model="gpt-4o", messages=limited_messages(), tools=tool_definitions()
                        ),
                    )
                    _accumulate_usage(response)
                    message = response.choices[0].message

            if message.content:
                console.print(f"[pink]  •[/pink] [white]{message.content}[/]")
                messages.append({"role": "assistant", "content": message.content})
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        # Display token usage summary
        console.print("\n[dim]Goodbye.[/dim]")
        if total_tokens > 0:
            console.print(
                Panel(
                    f"[bold]Token Usage Summary[/bold]\n"
                    f"Prompt tokens: [cyan]{total_prompt_tokens:,}[/cyan]\n"
                    f"Completion tokens: [cyan]{total_completion_tokens:,}[/cyan]\n"
                    f"Total tokens: [cyan]{total_tokens:,}[/cyan]",
                    border_style="dim",
                    title="Session Summary",
                )
            )
