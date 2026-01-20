import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from drift import paths
from drift.config import load_config

# Load environment variables from a local .env if present (for OPENAI_API_KEY, etc.).
load_dotenv()

# Work around Click 8.3+ API change where make_metavar(ctx) is required.
_original_make_metavar = click.core.Parameter.make_metavar


def _patched_make_metavar(self, ctx=None):  # type: ignore[override]
    return _original_make_metavar(self, ctx)


if _original_make_metavar.__code__.co_argcount == 2:
    click.core.Parameter.make_metavar = _patched_make_metavar  # type: ignore[assignment]

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def config(
    path_only: bool = typer.Option(
        False, "--path", "-p", is_flag=True, help="Print config path only."
    ),
) -> None:
    cfg_path = paths.config_path()
    load_config(cfg_path, create=True)
    typer.echo(str(cfg_path))
    if not path_only and cfg_path.exists():
        typer.echo(cfg_path.read_text(encoding="utf-8"))


@app.command()
def index(
    full: bool = typer.Option(False, "--full", help="Rebuild the index from scratch."),
) -> None:
    from drift.embeddings import build_image_embedder, build_text_embedder
    from drift.indexer import Indexer
    from drift.indexer.scan import (
        DEFAULT_ORGANIZED_SCORE_THRESHOLD,
        FolderAssessment,
        matches_expected_naming,
        scan_files,
    )
    from drift.indexer.types import FileInfo
    from drift.store import ChromaStore, MetadataStore

    config = load_config(create=True)
    console.print("[cyan]Scanning files...[/cyan]")
    files: list[FileInfo] = []
    git_skipped = 0
    scan_count = 0
    organized_dirs: list[FolderAssessment] = []
    needs_attention_dirs: list[FolderAssessment] = []
    git_repos: list[Path] = []

    def _record_assessment(assessment: FolderAssessment, status: str) -> None:
        if status == "organized":
            organized_dirs.append(assessment)
        else:
            needs_attention_dirs.append(assessment)

    def _record_git_repo(path: Path) -> None:
        git_repos.append(path)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as scan_progress:
        scan_task = scan_progress.add_task("Scanning...", total=None)
        for entry in scan_files(
            config,
            organized_threshold=DEFAULT_ORGANIZED_SCORE_THRESHOLD,
            on_directory=_record_assessment,
            on_git_repo=_record_git_repo,
        ):
            if entry is None:
                git_skipped += 1
                continue
            files.append(entry)
            scan_count += 1
            if scan_count % 200 == 0:
                scan_progress.update(
                    scan_task,
                    description=f"Scanning ({scan_count})… {Path(entry.path).parent}",
                )
        scan_progress.update(
            scan_task, description=f"Scanning complete ({scan_count})"
        )
    total = len(files)
    if total == 0:
        console.print("[yellow]No files found to index with current settings.[/yellow]")
        raise typer.Exit(code=0)

    type_counts: dict[str, int] = {}
    for f in files:
        type_counts[f.file_type.value] = type_counts.get(f.file_type.value, 0) + 1
    counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
    summary = f"[green]Found {total} files[/green]" + (f" ({counts_str})" if counts_str else "")
    if git_skipped:
        summary += f" [dim][skipped {git_skipped} git repos][/dim]"
    console.print(summary)

    metadata = MetadataStore(config.metadata_db_path())
    chroma = ChromaStore(config.chroma_dir())
    text_embedder = build_text_embedder(config)
    try:
        image_embedder = build_image_embedder(config)
    except RuntimeError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1) from exc

    root_loose_files = [
        Path(f.path)
        for f in files
        if Path(f.path).parent == config.scan.root
        and not matches_expected_naming(Path(f.path).name)
    ]

    indexer = Indexer(
        config=config,
        metadata=metadata,
        chroma=chroma,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Indexing", total=total)

        def on_progress(stats) -> None:
            desc = "Indexing"
            if stats.current:
                desc = f"Indexing: {Path(stats.current).name}"
            progress.update(task, completed=stats.scanned, description=desc)

        stats = indexer.run(full=full, files=files, progress=on_progress)
        progress.update(task, completed=total)
    metadata.close()
    stats.organized_dirs = organized_dirs
    stats.needs_attention_dirs = needs_attention_dirs

    table = Table(title="Index Summary")
    table.add_column("Scanned", justify="right")
    table.add_column("Indexed", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Removed", justify="right")
    table.add_row(
        str(stats.scanned),
        str(stats.indexed),
        str(stats.skipped),
        str(stats.failed),
        str(stats.removed),
    )
    console.print(table)

    def _render_folder_section(title: str, items: list[FolderAssessment], color: str, limit: int = 20) -> None:
        if not items:
            return
        extra = f" [dim](showing {min(len(items), limit)}/{len(items)})[/dim]" if len(items) > limit else ""
        console.print(f"[{color}]{title}[/]{extra}")
        for assessment in items[:limit]:
            console.print(
                f"- {assessment.path} (score {assessment.score:.2f}, files {assessment.files}, loose {assessment.loose}, subdirs {assessment.subdirs})"
            )

    if stats.organized_dirs or stats.needs_attention_dirs:
        console.print(
            f"\n[bold]Folder organization summary[/bold] "
            f"[dim](organized: {len(stats.organized_dirs)}, needs organization: {len(stats.needs_attention_dirs)})[/dim]"
        )
        organized_sorted = sorted(
            stats.organized_dirs, key=lambda a: (-a.score, str(a.path))
        )
        root_needs = []
        for a in stats.needs_attention_dirs:
            try:
                rel = Path(a.path).relative_to(config.scan.root)
            except Exception:
                continue
            if len(rel.parts) == 1:  # only direct children of the scan root
                root_needs.append(a)
        needs_sorted = sorted(
            root_needs, key=lambda a: (a.score, -a.files, str(a.path))
        )
        _render_folder_section(
            "Blacklisted as already organized",
            organized_sorted,
            "green",
        )
        _render_folder_section(
            "Needs organization (loose files detected)",
            needs_sorted,
            "yellow",
        )

    if root_loose_files:
        console.print("\n[magenta]Loose files at root (sorted scope)[/magenta]")
        limit = 40
        shown = 0
        for path in sorted(root_loose_files, key=lambda p: p.name)[:limit]:
            console.print(f"- {path.name} [dim]({path})[/dim]")
            shown += 1
        remaining = len(root_loose_files) - shown
        if remaining > 0:
            console.print(f"[dim]...and {remaining} more[/dim]")

    github_root = config.scan.root / "github"
    misplaced_repos = [
        p for p in git_repos if not p.is_relative_to(github_root)
    ]
    if misplaced_repos:
        console.print("\n[cyan]GitHub repos not under ~/github[/cyan]")
        for repo in sorted(misplaced_repos):
            console.print(f"- {repo}")
        console.print("[dim]Consider moving these into ~/github/<repo-name>[/dim]")
    console.print("[green]Indexing complete.[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, "--limit", min=1, max=50),
) -> None:
    from drift.embeddings import build_text_embedder
    from drift.store import ChromaStore, MetadataStore

    config = load_config(create=True)
    chroma = ChromaStore(config.chroma_dir())
    metadata = MetadataStore(config.metadata_db_path())
    text_embedder = build_text_embedder(config)
    embedding = text_embedder.embed_query(query)
    results = chroma.query_text(
        embedding, limit=limit, query=query, metadata=metadata
    )
    if not results:
        console.print("No results found.")
        return

    table = Table(title="Search Results")
    table.add_column("Path")
    table.add_column("Distance", justify="right")
    table.add_column("Snippet")
    for result in results:
        snippet = result.document[:160].replace("\n", " ")
        table.add_row(result.path, f"{result.distance:.4f}", snippet)
    console.print(table)


@app.command()
def chat() -> None:
    from drift.chat.loop import chat_loop

    chat_loop()


@app.command()
def structure(
    min_cluster_size: int = typer.Option(3, "--min-cluster-size", min=2, help="Minimum cluster size."),
    min_samples: int = typer.Option(2, "--min-samples", min=1, help="Minimum samples for clustering."),
) -> None:
    """Run suggest_structure directly and show the tree preview."""
    from drift.chat.tools.previews import display_structure_tree
    from drift.chat.tools import tool_suggest_structure
    from drift.store import ChromaStore

    config = load_config(create=True)
    client = OpenAI(timeout=60.0)
    chroma = ChromaStore(config.chroma_dir())
    console.print("[cyan]Analyzing file structure…[/cyan]")
    started = time.perf_counter()
    result = tool_suggest_structure(
        chroma=chroma,
        client=client,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        scan_root=config.scan.root,
        config=config,
    )
    elapsed = time.perf_counter() - started
    preview = result.content[:300] + ("…" if len(result.content) > 300 else "")
    console.print(f"[dim]Result preview: {preview}[/dim]")
    console.print(f"[dim]Analysis completed in {elapsed:.1f}s[/dim]")
    display_structure_tree(result.content, console=console, scan_root=config.scan.root)


@app.command()
def undo() -> None:
    """Undo the most recent applied move plan."""
    from drift.chat.tools.plan_ops import apply_move_plan
    from drift.undo import build_undo_plan, clear_last_action, load_last_action

    config = load_config(create=True)
    action = load_last_action()
    if not action or action.get("action") != "move":
        console.print("[yellow]No move action available to undo.[/yellow]")
        raise typer.Exit(code=1)

    scan_root = Path(action.get("scan_root") or config.scan.root)
    plan = build_undo_plan(action)
    if not plan:
        console.print("[yellow]Could not build undo plan.[/yellow]")
        raise typer.Exit(code=1)

    console.print("[cyan]Undo plan (first 20 moves):[/cyan]")
    for entry in plan[:20]:
        console.print(f"- {entry['src']} -> {entry['dest']}")
    if len(plan) > 20:
        console.print(f"[dim]...and {len(plan) - 20} more[/dim]")

    if not typer.confirm("Apply undo?"):
        console.print("[dim]Cancelled.[/dim]")
        raise typer.Exit(code=0)

    stats = apply_move_plan(plan, scan_root=scan_root)
    console.print(
        f"[green]Undo applied[/green] "
        f"(moved {stats['moved']}, "
        f"missing {stats['skipped_missing']}, "
        f"conflicts {stats['skipped_conflict']}, "
        f"outside-root {stats['skipped_outside']}, "
        f"errors {stats['errors']})."
    )
    clear_last_action()


def _display_startup_ascii_art() -> None:
    """Display DRIFT ASCII art banner when the program starts."""
    ascii_banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    ██████╗ ██████╗ ██╗███████╗████████╗                   ║
    ║    ██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝                   ║
    ║    ██║  ██║██████╔╝██║█████╗     ██║                      ║
    ║    ██║  ██║██╔══██╗██║██╔══╝     ██║                      ║
    ║    ██████╔╝██║  ██║██║██║        ██║                      ║
    ║    ╚══════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝                      ║
    ║                                                           ║
    ║    Organize your files with AI                            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(ascii_banner, style="cyan")
    console.print()


def main() -> None:
    if len(sys.argv) == 1:
        _display_startup_ascii_art()
        from drift.chat.loop import chat_loop

        chat_loop()
    else:
        app()
