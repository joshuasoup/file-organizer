<p align="center">
  <a href="https://github.com/joshuasoup/file-organizer">
    <style="font-size: 80px; margin: 0; text-align: center;">Drift</a>
  </a>
</p>
<p align="center">Organize your files with AI.</p>
<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square" /></a>
</p>

<p align="center">
  <a href="https://github.com/joshuasoup/file-organizer">
    <img src="terminal.png" alt="Drift Terminal UI" width="49%" />
  </a>
  <a href="https://github.com/joshuasoup/file-organizer">
    <img src="demo.gif" alt="Drift Demo" width="49%" />
  </a>
</p>

### Requirements

- macOS
- Python 3.12+ (3.12 recommended; Pillow wheels for 3.13 may lag)
- HuggingFace text embeddings (SentenceTransformers)
- CLIP deps for image embeddings: `torch` + `open_clip_torch`
- Semantic clustering: `hdbscan`
- OpenAI API key (for chat features)

### Setup

1. Create a virtualenv and install the package:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

2. Set your OpenAI API key (required for chat):

```bash
export OPENAI_API_KEY="your-api-key-here"
# or create a .env file with: OPENAI_API_KEY=your-api-key-here
```

### Configure

Generate the default config and inspect it:

```bash
drift config
```

Config file location:

```
~/Library/Application Support/drift/config.toml
```

You can override the config location (useful in sandboxes) with environment variables:

```
export drift_HOME="/path/to/writable/dir"
# or
export drift_CONFIG="/path/to/writable/dir/config.toml"
```

### Sandbox testing

Create an isolated sandbox with sample files (all left unsorted at root) and its own config (no real files touched):

```bash
python scripts/setup_sandbox.py            # or: python scripts/setup_sandbox.py /tmp/my-sandbox
export drift_CONFIG=$(pwd)/.sandbox_drift/config.toml  # path printed by the script
drift index --full
drift chat   # try search/structure too
```

Unset `drift_CONFIG` when you want to return to your real config.

Default behavior:

- Root is `~`
- Excludes `~/Applications`, `~/Music`, `~/Pictures`, `~/Library`, `~/Movies`, `~/.Trash`, `/Volumes/*`, `/Network/*`
- Skips macOS app bundles (`**/*.app`)
- Skips hidden files and symlinks
- Enforces file size limits per type
- Skips Git repos whose remotes include `github.com` (configurable)

You can edit `config.toml` to adjust ignore patterns, size limits, and models.

Text embeddings default to HuggingFace (`BAAI/bge-m3`). To switch to Ollama, set:

```
[embeddings]
text_provider = "ollama"
text_model = "bge-base-en-v1.5"
```

### Supported File Types

The indexer supports multiple file types with specialized extraction:

- **PDF**: Extracted using `pdfplumber`
- **DOCX**: Extracted using `python-docx`
- **Text files**: `.txt`, `.md`, `.markdown`, `.rtf`
- **Code files**: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.cpp`, `.c`, `.h`, `.hpp`, `.cs`, `.sh`, `.bash`, `.zsh`, `.sql`, `.html`, `.css`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tif`, `.tiff`, `.heic`, `.heif`, `.webp` (embedded using CLIP)

### Indexing

Run the indexer:

```bash
drift index
```

To rebuild from scratch:

```bash
drift index --full
```

The indexer will:

- Scan files and assess folder organization
- Extract text content from supported file types
- Generate embeddings for text and images
- Store vectors in ChromaDB and metadata in SQLite
- Report organized vs. needs-attention directories
- Show loose files at root level
- Identify misplaced GitHub repos

Data storage:

- Vectors: `~/Library/Application Support/drift/chroma/`
- Metadata: `~/Library/Application Support/drift/metadata.sqlite3`

### Search

Semantic search over indexed files:

```bash
drift search "find my tax receipts" [--limit 5]
```

### Chat

Interactive GPT-4o powered chat interface:

```bash
drift chat
```

Shortcut: running `drift` with no arguments also launches chat.

**Available Chat Tools:**

- **`search_files`**: Semantic search over indexed text and images
- **`find_duplicates`**: Find duplicate files by exact content hash
- **`suggest_structure`**: Suggest folder structure based on clustered embeddings (uses HDBSCAN + GPT-4o)
- **`preview_moves`**: Preview file move/rename plan before execution
- **`move_files`**: Move files with preview/approval flow
- **`delete_items`**: Delete specific files or folders (requires approval)
- **`undo_last_action`**: Undo the most recent applied move plan

The chat interface features:

- Auto-indexing on startup to pick up new files
- Automatic index refresh after moves/deletes
- Token usage tracking and summary
- Interactive previews for moves and deletes
- Structured tree display for organization suggestions

Type `exit` or `quit` to end the chat session.

### Structure Command

Run structure analysis directly from the command line:

```bash
drift structure [--min-cluster-size 3] [--min-samples 2]
```

This runs the `suggest_structure` tool and displays the proposed folder tree.

### Undo Command

Undo the most recent applied move plan:

```bash
drift undo
```

This will show a preview of the undo plan and ask for confirmation before applying.

### Manual testing checklist

- `drift config` creates the config file and shows defaults.
- `drift index` completes without errors and reports counts.
- `drift search "query"` returns relevant files.
- `drift chat` starts and responds to queries.
- `drift structure` generates folder organization suggestions.
- `drift undo` can reverse the last move operation.

### Troubleshooting

- If embeddings fail, ensure the HuggingFace model downloaded successfully
  (`SentenceTransformer("BAAI/bge-m3")`).
- If `pip install -e .` fails on Pillow with Python 3.13, upgrade Pillow
  (already loosened to `pillow>=10.3.0`) or use Python 3.12.
- If chat fails, ensure `OPENAI_API_KEY` is set in your environment or `.env` file.
- If structure suggestions fail, ensure `hdbscan` is installed (`pip install hdbscan`).
