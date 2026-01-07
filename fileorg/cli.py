import sys
from pathlib import Path

import typer
import click
from dotenv import load_dotenv
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

from fileorg import paths
from fileorg.config import load_config

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
    from fileorg.embeddings import build_image_embedder, build_text_embedder
    from fileorg.indexer import Indexer
    from fileorg.indexer.scan import scan_files
    from fileorg.store import ChromaStore, MetadataStore

    config = load_config(create=True)
    console.print("[cyan]Scanning files...[/cyan]")
    files = []
    git_skipped = 0
    scan_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as scan_progress:
        scan_task = scan_progress.add_task("Scanning...", total=None)
        for entry in scan_files(config):
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
    console.print("[green]Indexing complete.[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, "--limit", min=1, max=50),
) -> None:
    from fileorg.embeddings import build_text_embedder
    from fileorg.store import ChromaStore

    config = load_config(create=True)
    chroma = ChromaStore(config.chroma_dir())
    text_embedder = build_text_embedder(config)
    embedding = text_embedder.embed_query(query)
    results = chroma.query_text(embedding, limit=limit)
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
    from fileorg.chat.loop import chat_loop

    chat_loop()


def main() -> None:
    if len(sys.argv) == 1:
        from fileorg.chat.loop import chat_loop

        chat_loop()
    else:
        app()
