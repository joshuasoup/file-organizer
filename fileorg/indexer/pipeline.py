from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fileorg.config import AppConfig
from fileorg.embeddings import ImageEmbedder, TextEmbedder
from fileorg.indexer import chunk_text
from fileorg.indexer.extract import (
    extract_code_text,
    extract_docx_text,
    extract_pdf_text,
    extract_text,
    load_image,
)
from fileorg.indexer.scan import scan_files
from fileorg.indexer.types import FileInfo, FileType
from fileorg.store import ChromaStore, MetadataStore
from fileorg.utils import hash_file


@dataclass
class IndexStats:
    total: int = 0
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    current: str | None = None


ProgressCallback = Callable[[IndexStats], None]


class Indexer:
    def __init__(
        self,
        config: AppConfig,
        metadata: MetadataStore,
        chroma: ChromaStore,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
    ) -> None:
        self._config = config
        self._metadata = metadata
        self._chroma = chroma
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder

    def run(
        self,
        full: bool = False,
        files: list[FileInfo] | None = None,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        stats = IndexStats()
        if full:
            self._metadata.clear()
            self._chroma.clear()

        iterable = files if files is not None else scan_files(self._config)
        if files is not None:
            try:
                stats.total = len(files)
            except TypeError:
                stats.total = 0

        scanned_paths: set[str] = set()
        for info in iterable:
            stats.scanned += 1
            path_str = str(info.path)
            stats.current = path_str
            scanned_paths.add(path_str)

            if self._is_up_to_date(info):
                stats.skipped += 1
                if progress:
                    progress(stats)
                continue

            try:
                self._index_file(info)
                stats.indexed += 1
            except Exception:
                stats.failed += 1
            if progress:
                progress(stats)

        stats.removed = self._prune_missing(scanned_paths)
        stats.current = None
        if progress:
            progress(stats)
        return stats

    def _embedding_model_for(self, file_type: FileType) -> str:
        if file_type == FileType.IMAGE:
            return self._config.embeddings.image_model
        return self._config.embeddings.text_model

    def _is_up_to_date(self, info: FileInfo) -> bool:
        record = self._metadata.get_file(str(info.path))
        if record is None:
            return False
        if record.size != info.size or record.mtime != info.mtime:
            return False
        return record.embedding_model == self._embedding_model_for(info.file_type)

    def _index_file(self, info: FileInfo) -> None:
        path = info.path
        path_str = str(path)
        self._chroma.delete_path(path_str)

        if info.file_type == FileType.IMAGE:
            self._index_image(info)
        else:
            self._index_text(info)

        self._metadata.upsert_file(
            path=path_str,
            file_type=info.file_type.value,
            size=info.size,
            mtime=info.mtime,
            content_hash=hash_file(path),
            embedding_model=self._embedding_model_for(info.file_type),
            indexed_at=time.time(),
        )

    def _index_text(self, info: FileInfo) -> None:
        extractor = self._select_text_extractor(info.file_type)
        text = extractor(info.path)
        chunks = chunk_text(
            text,
            chunk_size=self._config.indexing.chunk_size,
            overlap=self._config.indexing.chunk_overlap,
            max_chunks=self._config.indexing.max_chunks_per_file,
        )
        if not chunks:
            return

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self._text_embedder.embed(chunk_texts)
        ids = [f"{info.path}:{index}" for index in range(len(chunks))]
        metadatas = [
            {"path": str(info.path), "chunk_index": index, "file_type": info.file_type.value}
            for index in range(len(chunks))
        ]
        self._chroma.add_text_embeddings(ids, embeddings, chunk_texts, metadatas)

    def _index_image(self, info: FileInfo) -> None:
        image = load_image(info.path)
        embeddings = self._image_embedder.embed([image])
        if not embeddings:
            return
        self._chroma.add_image_embeddings(
            ids=[f"{info.path}:image"],
            embeddings=embeddings,
            metadatas=[{"path": str(info.path), "file_type": info.file_type.value}],
        )

    def _select_text_extractor(self, file_type: FileType):
        if file_type == FileType.PDF:
            return extract_pdf_text
        if file_type == FileType.DOCX:
            return extract_docx_text
        if file_type == FileType.CODE:
            return extract_code_text
        return extract_text

    def _prune_missing(self, scanned_paths: set[str]) -> int:
        removed = 0
        for path in self._metadata.list_paths():
            if path in scanned_paths:
                continue
            self._metadata.delete_file(path)
            self._chroma.delete_path(path)
            removed += 1
        return removed
