from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from fileorg.chat.tools.plan_ops import (
    apply_delete_plan,
    apply_move_plan,
    build_move_plan,
    parse_move_plan,
    parse_structure_suggestions,
)
from fileorg.undo import clear_last_action, save_last_move


def display_search_results(
    content: str,
    console,
    query: str | None = None,
) -> None:
    """Display search results in a clean, user-friendly format."""
    try:
        results = json.loads(content)
    except Exception:
        return
    
    if not isinstance(results, list) or not results:
        return
    
    # Build a clean display
    panels = []
    for index, result in enumerate(results, 1):
        path_str = result.get("path", "")
        snippet = result.get("snippet", "")
        
        # Extract just the filename
        try:
            file_path = Path(path_str)
            filename = file_path.name
            # Show parent directory if it's meaningful and not too long
            parent = file_path.parent.name if file_path.parent.name and file_path.parent.name != "." else None
        except Exception:
            filename = path_str
            parent = None
        
        # Clean up snippet - remove extra whitespace and limit length
        snippet_clean = " ".join(snippet.split())
        if len(snippet_clean) > 180:
            snippet_clean = snippet_clean[:177] + "..."
        
        # Build the content
        content_lines = []
        if parent:
            content_lines.append(Text(f"📁 {parent}/", style="dim"))
        # Make filename clickable with file:// URL
        file_url = f"file://{path_str}"
        # Use Rich's markdown link syntax
        filename_link = f"[link={file_url}]{filename}[/link]"
        content_lines.append(Text.from_markup(filename_link, style="bold cyan"))
        # Add full path as clickable link below filename
        path_link = f"[link={file_url}]{path_str}[/link]"
        content_lines.append(Text.from_markup(path_link, style="dim"))
        if snippet_clean:
            content_lines.append(Text(""))
            content_lines.append(Text(snippet_clean, style="dim"))
        
        panel_content = Group(*content_lines)
        panels.append(
            Panel(
                panel_content,
                border_style="dim",
                padding=(0, 1),
            )
        )
    
    # Display all results with clean formatting
    console.print()
    if query:
        console.print(f"[bold]Found {len(results)} result{'s' if len(results) != 1 else ''} for '{query}':[/bold]\n")
    else:
        console.print(f"[bold]Found {len(results)} result{'s' if len(results) != 1 else ''}:[/bold]\n")
    
    for panel in panels:
        console.print(panel)
    console.print()


def display_duplicates(
    content: str,
    console,
) -> None:
    """Display duplicate files in a clean, user-friendly format."""
    try:
        duplicate_groups = json.loads(content)
    except Exception:
        return
    
    if not isinstance(duplicate_groups, list):
        return
    
    if not duplicate_groups:
        console.print()
        console.print("[bold green]No duplicate files found![/bold green]\n")
        return
    
    total_duplicates = sum(len(group) for group in duplicate_groups)
    total_groups = len(duplicate_groups)
    
    console.print()
    console.print(f"[bold]Found {total_groups} duplicate group{'s' if total_groups != 1 else ''} ({total_duplicates} total files):[/bold]\n")
    
    # Display only top 3 groups
    max_display = 3
    groups_to_show = duplicate_groups[:max_display]
    remaining = total_groups - len(groups_to_show)
    
    for group_index, group in enumerate(groups_to_show, 1):
        if not isinstance(group, list) or len(group) < 2:
            continue
        
        # Get file size (all files in group should have same size)
        file_size = group[0].get("size", 0) if group else 0
        size_str = f"{file_size:,} bytes" if file_size else "unknown size"
        
        console.print(f"[bold cyan]Group {group_index}[/bold cyan] - {len(group)} identical files ({size_str})")
        
        # Display each file in the group
        for file_info in group:
            path_str = file_info.get("path", "")
            if not path_str:
                continue
            
            # Extract filename and parent directory
            try:
                file_path = Path(path_str)
                filename = file_path.name
                parent = file_path.parent.name if file_path.parent.name and file_path.parent.name != "." else None
            except Exception:
                filename = path_str
                parent = None
            
            # Make path clickable
            file_url = f"file://{path_str}"
            path_link = f"[link={file_url}]{path_str}[/link]"
            
            # Display the line with clickable link
            if parent:
                console.print(f"  [dim]📁 {parent}/[/dim] [cyan link={file_url}]{path_str}[/cyan link={file_url}]")
            else:
                console.print(f"  [cyan link={file_url}]{path_str}[/cyan link={file_url}]")
        
        console.print()  # Empty line between groups
    
    # Show message if there are more groups
    if remaining > 0:
        console.print(f"[dim]... and {remaining} more duplicate group{'s' if remaining != 1 else ''}[/dim]\n")
    
    console.print()


def display_structure_tree(
    content: str,
    console,
    scan_root: Path | None = None,
    after_apply: Callable[[], None] | None = None,
) -> None:
    cleaned = content.lstrip()
    if not cleaned.startswith(("{", "[")):
        return
    suggestions = parse_structure_suggestions(content)
    if not suggestions:
        console.print("[yellow]Could not parse structure suggestions for tree preview.[/yellow]")
        return

    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Static, Tree
    except ImportError as exc:
        console.print(f"[yellow]Textual unavailable ({exc}); skipping structure preview.[/yellow]")
        return

    action_holder: dict[str, str | None] = {"value": None}

    class StructureTreeApp(App):
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("escape", "quit", "Quit"),
            ("enter", "quit", "Quit"),
            ("space", "quit", "Quit"),
            ("ctrl+c", "quit", "Quit"),
            ("a", "approve", "Approve"),
            ("c", "request_changes", "Request changes"),
            ("d", "decline", "Decline"),
        ]

        def __init__(self, data: list[dict[str, Any]]):
            super().__init__()
            self._data = data

        def compose(self) -> ComposeResult:
            tree = Tree("Suggested structure")
            tree.show_root = True
            tree.root.expand()

            path_nodes: dict[str, Any] = {}
            ordered = sorted(self._data, key=lambda s: str(s.get("path") or s.get("folder") or ""))

            for suggestion in ordered:
                path_str = suggestion.get("path") or suggestion.get("folder") or "cluster"
                parts = [p for p in str(path_str).split("/") if p]
                if not parts:
                    parts = [str(suggestion.get("folder", "cluster"))]

                parent = tree.root
                built: list[str] = []
                for part in parts:
                    built.append(part)
                    key = "/".join(built)
                    if key not in path_nodes:
                        path_nodes[key] = parent.add(part)
                    parent = path_nodes[key]

                parent.add(f"{suggestion.get('count', 0)} files")
                paths = suggestion.get("paths") or suggestion.get("sample_files") or []
                if paths:
                    for path in paths:
                        parent.add(str(path))
                else:
                    parent.add("[no files listed]")

            yield tree
            yield Static(
                "[bold]a[/] approve · [bold]c[/] request changes · [bold]d[/] decline · [bold]q/esc/enter/space[/] close",
                classes="actions",
            )

        def action_approve(self) -> None:  # type: ignore[override]
            action_holder["value"] = "approve"
            self.exit()

        def action_request_changes(self) -> None:  # type: ignore[override]
            action_holder["value"] = "request_changes"
            self.exit()

        def action_decline(self) -> None:  # type: ignore[override]
            action_holder["value"] = "decline"
            self.exit()

    console.print("[dim]Opening file tree preview (press q/esc/enter/space to close)…[/dim]")
    try:
        StructureTreeApp(suggestions).run(inline=True, inline_no_clear=True)
        action = action_holder.get("value")
        if action == "approve":
            plan = build_move_plan(suggestions, scan_root=scan_root)
            if not plan:
                console.print("[dim]No files to move in this plan.[/dim]")
            elif scan_root is None:
                console.print("[yellow]Cannot apply moves without a scan root. Showing preview only.[/yellow]")
            else:
                move_stats = apply_move_plan(plan, scan_root=scan_root)
                applied_moves = move_stats.get("applied") or []
                if applied_moves:
                    save_last_move(applied_moves, scan_root)
                else:
                    clear_last_action()
                console.print(
                    f"[green]Approved: applied move plan[/green] "
                    f"(moved {move_stats['moved']}, "
                    f"missing {move_stats['skipped_missing']}, "
                    f"conflicts {move_stats['skipped_conflict']}, "
                    f"outside-root {move_stats['skipped_outside']}, "
                    f"errors {move_stats['errors']})."
                )
                if after_apply:
                    after_apply()
                if plan:
                    console.print(f"[bold]Moves ({min(len(plan), 30)}/{len(plan)} shown):[/bold]")
                    for entry in plan[:30]:
                        console.print(f"- {entry['src']} -> {entry['dest']}")
                    if len(plan) > 30:
                        console.print(f"[dim]...and {len(plan) - 30} more[/dim]")
        elif action == "request_changes":
            console.print("[cyan]Request changes: describe adjustments and re-run structuring.[/cyan]")
            note = console.input("[dim]Enter change notes (or leave blank to skip): [/dim]")
            if note.strip():
                console.print(f"[dim]Noted change request:[/dim] {note}")
        elif action == "decline":
            console.print("[yellow]Declined: do not apply these moves.[/yellow]")
        else:
            console.print(
                "[dim]No action selected. You can approve (a), request changes (c), or decline (d).[/dim]"
            )
    except KeyboardInterrupt:
        console.print("[yellow]Closed preview (Ctrl+C).[/yellow]")
    except Exception as exc:
        console.print(f"[yellow]Couldn't open Textual tree ({exc}); skipping preview.[/yellow]")


def display_move_preview(
    content: str,
    console,
    scan_root: Path | None = None,
    after_apply: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    plan = parse_move_plan(content)
    if not plan:
        console.print("[yellow]Could not parse move plan.[/yellow]")
        return None

    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Static, Tree
    except ImportError as exc:
        console.print(f"[yellow]Textual unavailable ({exc}); skipping move preview.[/yellow]")
        return None

    action_holder: dict[str, str | None] = {"value": None}
    stats_holder: dict[str, int] | None = None

    class MovePlanApp(App):
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("escape", "quit", "Quit"),
            ("enter", "quit", "Quit"),
            ("space", "quit", "Quit"),
            ("a", "approve", "Approve"),
            ("c", "request_changes", "Request changes"),
            ("d", "decline", "Decline"),
        ]

        def __init__(self, data: list[dict[str, str]], scan_root: Path | None) -> None:
            super().__init__()
            self._data = data
            self._scan_root = scan_root

        def compose(self) -> ComposeResult:
            tree = Tree("Move plan")
            tree.show_root = True
            tree.root.expand()

            path_nodes: dict[str, Any] = {}
            for entry in self._data:
                raw_dest = entry.get("dest", "")
                src = entry.get("src", "")
                try:
                    dest_path = Path(raw_dest)
                    if self._scan_root:
                        try:
                            dest_path = dest_path.resolve().relative_to(self._scan_root.resolve())
                        except Exception:
                            dest_path = dest_path
                    parts = [p for p in dest_path.parts if p]
                except Exception:
                    parts = [raw_dest] if raw_dest else []

                parent = tree.root
                built: list[str] = []
                for idx, part in enumerate(parts):
                    built.append(part)
                    key = "/".join(built)
                    if key not in path_nodes:
                        path_nodes[key] = parent.add(part)
                    parent = path_nodes[key]
                    if idx == len(parts) - 1:
                        parent.add(src or "[no src]")

            yield tree
            yield Static(
                "[bold]a[/] approve · [bold]c[/] request changes · [bold]d[/] decline · [bold]q/esc/enter/space[/] close",
                classes="actions",
            )

        def action_approve(self) -> None:  # type: ignore[override]
            action_holder["value"] = "approve"
            self.exit()

        def action_request_changes(self) -> None:  # type: ignore[override]
            action_holder["value"] = "request_changes"
            self.exit()

        def action_decline(self) -> None:  # type: ignore[override]
            action_holder["value"] = "decline"
            self.exit()

    console.print("[dim]Opening move plan preview (press q/esc/enter/space to close)…[/dim]")
    try:
        MovePlanApp(plan, scan_root=scan_root).run(inline=True, inline_no_clear=True)
        action = action_holder.get("value")
        if action == "approve":
            if scan_root is None:
                console.print("[yellow]Cannot apply moves without a scan root. Preview only.[/yellow]")
            else:
                move_stats = apply_move_plan(plan, scan_root=scan_root)
                applied_moves = move_stats.get("applied") or []
                if applied_moves:
                    save_last_move(applied_moves, scan_root)
                else:
                    clear_last_action()
                console.print(
                    f"[green]Approved: applied move plan[/green] "
                    f"(moved {move_stats['moved']}, "
                    f"missing {move_stats['skipped_missing']}, "
                    f"conflicts {move_stats['skipped_conflict']}, "
                    f"outside-root {move_stats['skipped_outside']}, "
                    f"errors {move_stats['errors']})."
                )
                if after_apply:
                    after_apply()
                stats_holder = move_stats
        elif action == "request_changes":
            console.print("[cyan]Request changes: describe adjustments and re-run preview.[/cyan]")
            note = console.input("[dim]Enter change notes (or leave blank to skip): [/dim]")
            if note.strip():
                console.print(f"[dim]Noted change request:[/dim] {note}")
        elif action == "decline":
            console.print("[yellow]Declined: do not apply these moves.[/yellow]")
        else:
            console.print(
                "[dim]No action selected. You can approve (a), request changes (c), or decline (d).[/dim]"
            )
    except KeyboardInterrupt:
        console.print("[yellow]Closed preview (Ctrl+C).[/yellow]")
        return {"action": None, "stats": stats_holder, "plan_size": len(plan), "plan": plan}
    except Exception as exc:
        console.print(f"[yellow]Couldn't open Textual move preview ({exc}); skipping preview.[/yellow]")
    return {
        "action": action_holder.get("value"),
        "stats": stats_holder,
        "plan_size": len(plan),
        "plan": plan,
    }


def display_delete_preview(
    content: str,
    console,
    scan_root: Path | None = None,
    after_apply: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    try:
        items = json.loads(content)
    except Exception:
        console.print("[yellow]Could not parse delete plan.[/yellow]")
        return None
    if not isinstance(items, list):
        console.print("[yellow]Delete plan must be a list.[/yellow]")
        return None

    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Static, Tree
    except ImportError as exc:
        console.print(f"[yellow]Textual unavailable ({exc}); skipping delete preview.[/yellow]")
        return None

    action_holder: dict[str, str | None] = {"value": None}
    stats_holder: dict[str, int] | None = None

    class DeletePlanApp(App):
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("escape", "quit", "Quit"),
            ("enter", "quit", "Quit"),
            ("space", "quit", "Quit"),
            ("a", "approve", "Approve"),
            ("c", "request_changes", "Request changes"),
            ("d", "decline", "Decline"),
        ]

        def __init__(self, data: list[str], scan_root: Path | None) -> None:
            super().__init__()
            self._data = data
            self._scan_root = scan_root

        def compose(self) -> ComposeResult:
            tree = Tree("Delete plan")
            tree.show_root = True
            tree.root.expand()

            path_nodes: dict[str, Any] = {}
            for raw in self._data:
                try:
                    candidate = Path(raw)
                    if self._scan_root:
                        try:
                            candidate = candidate.resolve().relative_to(self._scan_root.resolve())
                        except Exception:
                            candidate = candidate
                    parts = [p for p in candidate.parts if p]
                except Exception:
                    parts = [raw] if raw else []

                parent = tree.root
                built: list[str] = []
                for idx, part in enumerate(parts):
                    built.append(part)
                    key = "/".join(built)
                    if key not in path_nodes:
                        path_nodes[key] = parent.add(part)
                    parent = path_nodes[key]
                    if idx == len(parts) - 1:
                        parent.add("[delete]")

            yield tree
            yield Static(
                "[bold]a[/] approve · [bold]c[/] request changes · [bold]d[/] decline · [bold]q/esc/enter/space[/] close",
                classes="actions",
            )

        def action_approve(self) -> None:  # type: ignore[override]
            action_holder["value"] = "approve"
            self.exit()

        def action_request_changes(self) -> None:  # type: ignore[override]
            action_holder["value"] = "request_changes"
            self.exit()

        def action_decline(self) -> None:  # type: ignore[override]
            action_holder["value"] = "decline"
            self.exit()

    console.print("[dim]Opening delete plan preview (press q/esc/enter/space to close)…[/dim]")
    try:
        DeletePlanApp(items, scan_root=scan_root).run(inline=True, inline_no_clear=True)
        action = action_holder.get("value")
        if action == "approve":
            if scan_root is None:
                console.print("[yellow]Cannot delete without a scan root. Preview only.[/yellow]")
            else:
                delete_stats = apply_delete_plan(items, scan_root=scan_root)
                console.print(
                    f"[red]Approved: deleted items[/red] "
                    f"(deleted {delete_stats['deleted']}, "
                    f"missing {delete_stats['skipped_missing']}, "
                    f"outside-root {delete_stats['skipped_outside']}, "
                    f"errors {delete_stats['errors']})."
                )
                if after_apply:
                    after_apply()
                stats_holder = delete_stats
        elif action == "request_changes":
            console.print("[cyan]Request changes: describe adjustments and re-run preview.[/cyan]")
            note = console.input("[dim]Enter change notes (or leave blank to skip): [/dim]")
            if note.strip():
                console.print(f"[dim]Noted change request:[/dim] {note}")
        elif action == "decline":
            console.print("[yellow]Declined: do not delete these items.[/yellow]")
        else:
            console.print(
                "[dim]No action selected. You can approve (a), request changes (c), or decline (d).[/dim]"
            )
    except KeyboardInterrupt:
        console.print("[yellow]Closed preview (Ctrl+C).[/yellow]")
        return {"action": None, "stats": stats_holder, "plan_size": len(items)}
    except Exception as exc:
        console.print(f"[yellow]Couldn't open Textual delete preview ({exc}); skipping preview.[/yellow]")
    return {"action": action_holder.get("value"), "stats": stats_holder, "plan_size": len(items)}
