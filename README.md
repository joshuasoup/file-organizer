## FileOrg (MVP)

Local-first file indexer + semantic search for macOS.

### Requirements

- macOS
- Python 3.11+ (3.11 or 3.12 recommended; Pillow wheels for 3.13 may lag)
- HuggingFace text embeddings (SentenceTransformers)
- CLIP deps for image embeddings: `torch` + `open_clip_torch`
- Semantic clustering: `hdbscan`

### Setup

1. Create a virtualenv and install the package:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

2. Install CLIP dependencies (required for image indexing):

```bash
pip install torch open_clip_torch
```

3. (One-time) Let SentenceTransformers download the text embedding model (offline after first pull):

```python
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-base-en-v1.5")
print("Model downloaded.")
PY
```

4. (Optional) Ensure `hdbscan` is installed for semantic folder suggestions (pulled via `pip install -e .`):

```bash
pip install hdbscan
```

### Configure

Generate the default config and inspect it:

```bash
fileorg config
```

Config file location:

```
~/Library/Application Support/FileOrg/config.toml
```

You can override the config location (useful in sandboxes) with environment variables:

```
export FILEORG_HOME="/path/to/writable/dir"
# or
export FILEORG_CONFIG="/path/to/writable/dir/config.toml"
```

Default behavior:

- Root is `~`
- Excludes `~/Applications`, `~/Music`, `~/Pictures`, `~/Library`, `~/Movies`, `~/.Trash`, `/Volumes/*`, `/Network/*`
- Skips macOS app bundles (`**/*.app`)
- Skips hidden files and symlinks
- Enforces file size limits per type
- Skips Git repos whose remotes include `github.com` (configurable)

You can edit `config.toml` to adjust ignore patterns, size limits, and models.

Text embeddings default to HuggingFace (`BAAI/bge-base-en-v1.5`). To switch back to Ollama, set:

```
[embeddings]
text_provider = "ollama"
text_model = "bge-base-en-v1.5"
```

### Indexing

Run the indexer:

```bash
fileorg index
```

To rebuild from scratch:

```bash
fileorg index --full
```

Data storage:

- Vectors: `~/Library/Application Support/FileOrg/chroma/`
- Metadata: `~/Library/Application Support/FileOrg/metadata.sqlite3`

### Search

```bash
fileorg search "find my tax receipts"
```

### Chat

```bash
fileorg chat
```

This opens a GPT-4o powered chat with tools for semantic search, duplicates, stale files, structure suggestions, and move previews. Type `exit` to quit. Requires `OPENAI_API_KEY`.

Shortcut: running `fileorg` with no arguments also launches chat.

The “suggest structure” tool clusters embeddings (text + images) with HDBSCAN and asks GPT-4o to name the clusters. If `hdbscan` isn’t installed, the tool will say so.

### Manual testing checklist

- `fileorg config` creates the config file and shows defaults.
- `fileorg index` completes without errors and reports counts.
- `fileorg search "query"` returns relevant files.

### Troubleshooting

- If `fileorg index` fails with a CLIP error, install deps:
  `pip install torch open_clip_torch`
- If embeddings fail, ensure the HuggingFace model downloaded successfully
  (`SentenceTransformer("BAAI/bge-base-en-v1.5")`).
- If structure suggestions fail, install `hdbscan` and re-run indexing if needed.
- If `pip install -e .` fails on Pillow with Python 3.13, upgrade Pillow
  (already loosened to `pillow>=10.3.0`) or use Python 3.12.
