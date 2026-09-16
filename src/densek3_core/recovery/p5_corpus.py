"""Deterministic document-split-first corpus materialization for P5.3-Probe."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

P5_SEQUENCE_LENGTH = 2048
P5_SPLIT_SEQUENCES = {"train": 976, "dev": 97, "heldout": 97}
P5_PROBE_RANGES = {
    "A": {"start_sequence": 0, "end_sequence": 244, "effective_tokens": 499_712},
    "B": {"start_sequence": 244, "end_sequence": 488, "effective_tokens": 499_712},
    "C": {"start_sequence": 488, "end_sequence": 976, "effective_tokens": 999_424},
}
P5_SPLIT_SEED = "DenseK3-P5-Probe-corpus-v1"
UINT32_MAX = 2**32 - 1


def normalize_document(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_data_asset(data_dir: Path, declared: str | Path) -> Path:
    path = Path(declared)
    candidates = [path] if path.is_absolute() else [data_dir / path, data_dir / "raw" / path]
    if not path.is_absolute():
        candidates.append(data_dir.parents[1] / path)
        parts = path.parts
        if len(parts) >= 2 and parts[:2] == ("data", "p5-probe"):
            candidates.append(data_dir.joinpath(*parts[2:]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.is_relative_to(data_dir):
            return resolved
    raise FileNotFoundError(f"P5 corpus asset not found inside {data_dir}: {declared}")


def _manifest_hash_records(value: Any) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        path = next(
            (
                value.get(key)
                for key in ("path", "file", "filename", "local_path", "relative_path")
                if value.get(key)
            ),
            None,
        )
        digest = next(
            (value.get(key) for key in ("sha256", "sha256sum", "digest") if value.get(key)),
            None,
        )
        if path is not None and isinstance(digest, str) and len(digest) == 64:
            yield str(path), digest.lower()
        for child in value.values():
            yield from _manifest_hash_records(child)
        for key, child in value.items():
            if isinstance(child, dict) and any(
                isinstance(child.get(digest_key), str)
                for digest_key in ("sha256", "sha256sum", "digest")
            ):
                digest = next(
                    child[digest_key]
                    for digest_key in ("sha256", "sha256sum", "digest")
                    if isinstance(child.get(digest_key), str)
                )
                if len(digest) == 64 and any(
                    marker in str(key).lower() for marker in (".parquet", ".jsonl", ".json")
                ):
                    yield str(key), digest.lower()
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_hash_records(child)


def verify_download_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Verify every SHA256-addressed file declared by the download manifest."""
    data_dir = Path(data_dir).resolve()
    path = data_dir / "download-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = set(_manifest_hash_records(manifest))
    named_manifests = {
        "source_tree_sha256": "source-tree.jsonl",
        "source_shards_manifest_sha256": "source-shards.jsonl",
        "candidate_documents_manifest_sha256": "candidate-documents.jsonl",
    }
    for key, filename in named_manifests.items():
        digest = manifest.get(key)
        if isinstance(digest, str) and len(digest) == 64:
            records.add((filename, digest.lower()))
    source_shards_path = data_dir / "source-shards.jsonl"
    if source_shards_path.is_file():
        for line in source_shards_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            declared = record.get("local_path", record.get("relative_path"))
            digest = record.get("sha256")
            if declared and isinstance(digest, str) and len(digest) == 64:
                records.add((str(declared), digest.lower()))
    records = sorted(records)
    if not records:
        raise ValueError("download-manifest.json contains no recognizable path/SHA256 records")
    results = []
    for declared, expected in records:
        candidate = _resolve_data_asset(data_dir, declared)
        observed = sha256_file(candidate)
        results.append(
            {
                "path": candidate.relative_to(data_dir).as_posix(),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": observed == expected,
            }
        )
    if not all(item["passed"] for item in results):
        raise ValueError("One or more downloaded P5 corpus assets failed SHA256 verification")
    return {
        "manifest_path": path.relative_to(data_dir).as_posix(),
        "manifest_sha256": sha256_file(path),
        "files": results,
        "all_sha256_verified": True,
    }


class ParquetTextReader:
    """Resolve selected Parquet rows without loading a complete 10BT shard."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._files: dict[Path, Any] = {}

    @staticmethod
    def _dependencies():
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet-backed P5 candidates require pyarrow") from exc
        return pq

    def _path(self, value: str) -> Path:
        return _resolve_data_asset(self.data_dir, value)

    def _file(self, path: Path):
        if path not in self._files:
            self._files[path] = self._dependencies().ParquetFile(path)
        return self._files[path]

    def _row_group_for_global_index(self, parquet: Any, row_index: int) -> tuple[int, int]:
        if row_index < 0 or row_index >= parquet.metadata.num_rows:
            raise IndexError(f"Parquet row index out of range: {row_index}")
        remaining = row_index
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if remaining < count:
                return group, remaining
            remaining -= count
        raise AssertionError("Parquet row-group lookup did not terminate")

    def _location(self, record: dict[str, Any]) -> tuple[Path, int, int, str]:
        declared = next(
            (
                record.get(key)
                for key in (
                    "parquet_path",
                    "parquet_file",
                    "source_file",
                    "source_path",
                    "shard",
                    "file",
                    "path",
                )
                if record.get(key)
            ),
            None,
        )
        if declared is None:
            raise ValueError("Candidate record has neither embedded text nor a Parquet path")
        path = self._path(str(declared))
        parquet = self._file(path)
        text_column = str(record.get("text_column", "text"))
        if "row_group" in record and any(key in record for key in ("row_in_group", "row_offset")):
            group = int(record["row_group"])
            row = int(record.get("row_in_group", record.get("row_offset")))
        elif any(
            key in record
            for key in ("row_index", "source_row_index", "source_row", "parquet_row", "row", "index")
        ):
            global_row = int(
                next(
                    record[key]
                    for key in (
                        "row_index",
                        "source_row_index",
                        "source_row",
                        "parquet_row",
                        "row",
                        "index",
                    )
                    if key in record
                )
            )
            group, row = self._row_group_for_global_index(parquet, global_row)
        else:
            raise ValueError("Parquet candidate must define row_index or row_group plus row_in_group")
        return path, group, row, text_column

    def read_many(self, records: list[CandidateDocument]) -> dict[str, tuple[str, dict[str, Any]]]:
        """Read selected rows grouped by row group, without retaining the 2GB shard in memory."""
        groups: dict[tuple[Path, int, str], list[tuple[str, int]]] = defaultdict(list)
        results: dict[str, tuple[str, dict[str, Any]]] = {}
        for document in records:
            if document.text is not None:
                results[document.document_id] = (document.text, document.source)
                continue
            path, group, row, text_column = self._location(document.source)
            groups[(path, group, text_column)].append((document.document_id, row))
        for (path, group, text_column), selected in groups.items():
            table = self._file(path).read_row_group(group, columns=[text_column])
            for document_id, row in selected:
                if row < 0 or row >= table.num_rows:
                    raise IndexError(f"Parquet row-in-group out of range: {row}")
                text = table.column(text_column)[row].as_py()
                if not isinstance(text, str):
                    raise TypeError(f"Parquet candidate text is not a string: {path}:{group}:{row}")
                text = normalize_document(text)
                results[document_id] = (
                    text,
                    {
                        "parquet_path": path.relative_to(self.data_dir).as_posix(),
                        "row_group": group,
                        "row_in_group": row,
                        "text_column": text_column,
                    },
                )
        return results


@dataclass(frozen=True)
class CandidateDocument:
    document_id: str
    text: str | None
    text_sha256: str
    source: dict[str, Any]
    membership_hash: str
    split: str
    expected_token_count_including_eos: int


def _membership(document_id: str, text_sha256: str) -> tuple[str, str]:
    key = f"{P5_SPLIT_SEED}\0{document_id}\0{text_sha256}".encode()
    digest = hashlib.sha256(key).hexdigest()
    bucket = int(digest[:16], 16) % 10_000
    if bucket < 8_000:
        split = "train"
    elif bucket < 9_000:
        split = "dev"
    else:
        split = "heldout"
    return digest, split


def load_candidate_documents(
    data_dir: str | Path,
    tokenizer: Any,
) -> tuple[list[CandidateDocument], dict[str, Any]]:
    """Globally deduplicate by frozen rank, then select enough documents per split."""
    data_dir = Path(data_dir).resolve()
    path = data_dir / "candidate-documents.jsonl"
    parquet = ParquetTextReader(data_dir)
    records_seen = 0
    empty = 0
    schema_counts: dict[str, int] = {"embedded_text": 0, "parquet_reference": 0}
    selected_token_totals = {split: 0 for split in P5_SPLIT_SEQUENCES}
    with tempfile.TemporaryDirectory(prefix="p5-candidate-index-", dir=data_dir) as temporary:
        database_path = Path(temporary) / "candidates.sqlite3"
        connection = sqlite3.connect(database_path)
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE raw_candidates (
                candidate_line INTEGER PRIMARY KEY,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                document_id TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                membership_hash TEXT NOT NULL,
                split TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                schema_kind TEXT NOT NULL
            );
            """
        )
        batch = []
        with path.open("rb") as handle:
            line_number = 0
            while raw_line := handle.readline():
                line_number += 1
                if not raw_line.strip():
                    continue
                byte_offset = handle.tell() - len(raw_line)
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise TypeError(f"Candidate line {line_number} is not a JSON object")
                embedded = next(
                    (
                        record.get(key)
                        for key in ("text", "content", "document")
                        if isinstance(record.get(key), str)
                    ),
                    None,
                )
                text = None if embedded is None else normalize_document(embedded)
                schema_kind = "parquet_reference" if text is None else "embedded_text"
                schema_counts[schema_kind] += 1
                if text == "":
                    empty += 1
                    continue
                declared_text_hash = record.get("text_sha256")
                if text is not None:
                    observed_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if declared_text_hash is not None and declared_text_hash != observed_text_hash:
                        raise ValueError(f"Candidate line {line_number} embedded text SHA256 mismatch")
                    text_hash = observed_text_hash
                elif not isinstance(declared_text_hash, str) or len(declared_text_hash) != 64:
                    raise ValueError(f"Candidate line {line_number} lacks a valid text_sha256")
                else:
                    text_hash = declared_text_hash.lower()
                try:
                    int(text_hash, 16)
                except ValueError as exc:
                    raise ValueError(f"Candidate line {line_number} has a non-hex text SHA256") from exc
                declared_id = next(
                    (
                        record.get(key)
                        for key in ("document_id", "source_document_id", "id", "doc_id")
                        if record.get(key) is not None
                    ),
                    None,
                )
                document_id = str(declared_id) if declared_id is not None else text_hash
                membership_hash, hashed_split = _membership(document_id, text_hash)
                declared_rank = record.get("doc_rank")
                if declared_rank is not None:
                    if not isinstance(declared_rank, str) or len(declared_rank) != 64:
                        raise ValueError(f"Candidate line {line_number} has an invalid doc_rank")
                    try:
                        int(declared_rank, 16)
                    except ValueError as exc:
                        raise ValueError(f"Candidate line {line_number} has a non-hex doc_rank") from exc
                    membership_hash = declared_rank.lower()
                declared_split = record.get("split")
                if declared_split is not None and declared_split not in P5_SPLIT_SEQUENCES:
                    raise ValueError(f"Candidate line {line_number} has invalid split: {declared_split}")
                split = hashed_split if declared_split is None else str(declared_split)
                expected_tokens = record.get("qwen_token_count_including_eos")
                if expected_tokens is None:
                    if text is None:
                        raise ValueError(f"Candidate line {line_number} lacks a frozen token count")
                    expected_tokens = len(tokenizer.encode(text, add_special_tokens=False)) + 1
                expected_tokens = int(expected_tokens)
                if expected_tokens <= 0:
                    raise ValueError(f"Candidate line {line_number} has a non-positive token count")
                batch.append(
                    (
                        line_number,
                        byte_offset,
                        len(raw_line),
                        document_id,
                        text_hash,
                        membership_hash,
                        split,
                        expected_tokens,
                        schema_kind,
                    )
                )
                records_seen += 1
                if len(batch) == 10_000:
                    connection.executemany(
                        "INSERT INTO raw_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO raw_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
        connection.executescript(
            """
            CREATE INDEX raw_candidates_rank
            ON raw_candidates(membership_hash, document_id, candidate_line);
            """
        )
        connection.commit()
        duplicate_id_occurrences = records_seen - int(
            connection.execute("SELECT COUNT(DISTINCT document_id) FROM raw_candidates").fetchone()[0]
        )
        duplicate_text_occurrences = records_seen - int(
            connection.execute("SELECT COUNT(DISTINCT text_sha256) FROM raw_candidates").fetchone()[0]
        )
        selected_rows = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        selected_complete = set()
        rows = connection.execute(
            """
            SELECT candidate_line, byte_offset, byte_length, document_id, text_sha256,
                   membership_hash, split, token_count, schema_kind
            FROM raw_candidates
            ORDER BY membership_hash, document_id, candidate_line
            """
        )
        for row in rows:
            document_id, text_hash, split = str(row[3]), str(row[4]), str(row[6])
            if document_id in seen_ids or text_hash in seen_texts:
                continue
            seen_ids.add(document_id)
            seen_texts.add(text_hash)
            if split not in selected_complete:
                selected_rows.append(row)
                selected_token_totals[split] += int(row[7])
                target = P5_SPLIT_SEQUENCES[split] * P5_SEQUENCE_LENGTH
                if selected_token_totals[split] >= target:
                    selected_complete.add(split)
        unique_count = len(seen_ids)
        for split, sequence_count in P5_SPLIT_SEQUENCES.items():
            target = sequence_count * P5_SEQUENCE_LENGTH
            if selected_token_totals[split] < target:
                raise ValueError(f"Candidate split {split} has insufficient unique frozen tokens")
        connection.close()

        selected = []
        with path.open("rb") as handle:
            for row in selected_rows:
                line_number, byte_offset, byte_length = (int(row[0]), int(row[1]), int(row[2]))
                handle.seek(byte_offset)
                raw_line = handle.read(byte_length)
                record = json.loads(raw_line)
                embedded = next(
                    (
                        record.get(key)
                        for key in ("text", "content", "document")
                        if isinstance(record.get(key), str)
                    ),
                    None,
                )
                text = None if embedded is None else normalize_document(embedded)
                source = dict(record) if text is None else {}
                source.update({"candidate_line": line_number, "kind": str(row[8])})
                selected.append(
                    CandidateDocument(
                        str(row[3]),
                        text,
                        str(row[4]),
                        source,
                        str(row[5]),
                        str(row[6]),
                        int(row[7]),
                    )
                )
    if not selected:
        raise ValueError("No usable P5 candidate documents were resolved")
    resolved = parquet.read_many(selected)
    documents = []
    for document in selected:
        text, resolved_source = resolved[document.document_id]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash != document.text_sha256:
            raise ValueError(f"Selected document text SHA256 mismatch: {document.document_id}")
        token_count = len(tokenizer.encode(text, add_special_tokens=False)) + 1
        if token_count != document.expected_token_count_including_eos:
            raise ValueError(
                f"Selected document token count mismatch: {document.document_id}: "
                f"expected {document.expected_token_count_including_eos}, observed {token_count}"
            )
        documents.append(
            replace(document, text=text, source={**document.source, **resolved_source})
        )
    if len({item.document_id for item in documents}) != len(documents) or len(
        {item.text_sha256 for item in documents}
    ) != len(documents):
        raise AssertionError("Global candidate deduplication did not produce unique selected documents")
    documents.sort(key=lambda item: (item.membership_hash, item.document_id))
    return documents, {
        "candidate_path": path.relative_to(data_dir).as_posix(),
        "candidate_sha256": sha256_file(path),
        "candidate_records_scanned": records_seen,
        "unique_candidate_documents": unique_count,
        "resolved_documents": len(documents),
        "selected_tokens_including_eos": selected_token_totals,
        "duplicate_documents_excluded": records_seen - unique_count,
        "duplicate_id_occurrences": duplicate_id_occurrences,
        "duplicate_text_occurrences": duplicate_text_occurrences,
        "empty_documents_excluded": empty,
        "schema_counts": schema_counts,
        "membership_algorithm": (
            "global greedy uniqueness by ascending (doc_rank, document_id, candidate_line), "
            "then frozen explicit split selection and deterministic packing; hash fallback"
        ),
        "membership_seed": P5_SPLIT_SEED,
    }


def tokenizer_identity(tokenizer: Any, tokenizer_dir: str | Path) -> dict[str, Any]:
    tokenizer_dir = Path(tokenizer_dir).resolve()
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    )
    files = {
        name: sha256_file(tokenizer_dir / name)
        for name in names
        if (tokenizer_dir / name).is_file()
    }
    if not files:
        raise FileNotFoundError("P5 tokenizer directory contains no recognized tokenizer files")
    if tokenizer.eos_token_id is None:
        raise ValueError("P5 packing requires a frozen EOS token ID")
    identity = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": int(tokenizer.eos_token_id),
        "special_tokens_map": tokenizer.special_tokens_map,
        "files": files,
    }
    identity["identity_hash"] = canonical_json_hash(identity)
    return identity


def _write_uint32(path: Path, values: Iterable[int]) -> None:
    array = np.asarray(list(values), dtype="<u4")
    path.write_bytes(array.tobytes(order="C"))


def _write_index(path: Path, sequence_count: int, sequence_length: int) -> None:
    offsets = np.arange(sequence_count + 1, dtype="<u8") * sequence_length
    path.write_bytes(offsets.tobytes(order="C"))


def _materialize_once(
    documents: list[CandidateDocument],
    tokenizer: Any,
    tokenizer_report: dict[str, Any],
    output_dir: Path,
    source_report: dict[str, Any],
    download_report: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    eos = int(tokenizer.eos_token_id)
    split_reports: dict[str, Any] = {}
    split_memberships: dict[str, set[str]] = {}
    for split, sequence_count in P5_SPLIT_SEQUENCES.items():
        target_tokens = sequence_count * P5_SEQUENCE_LENGTH
        stream: list[int] = []
        memberships = []
        eos_positions = []
        for document in (item for item in documents if item.split == split):
            if document.text is None:
                raise ValueError(f"Selected document text was not resolved: {document.document_id}")
            token_ids = tokenizer.encode(document.text, add_special_tokens=False)
            if any(token < 0 or token > UINT32_MAX for token in token_ids):
                raise ValueError("Tokenizer emitted an ID outside uint32 range")
            start = len(stream)
            stream.extend(int(token) for token in token_ids)
            eos_position = len(stream)
            stream.append(eos)
            eos_positions.append(eos_position)
            memberships.append(
                {
                    "document_id": document.document_id,
                    "text_sha256": document.text_sha256,
                    "membership_hash": document.membership_hash,
                    "source": document.source,
                    "token_start": start,
                    "token_count_without_eos": len(token_ids),
                    "eos_position": eos_position,
                }
            )
            if len(stream) >= target_tokens:
                break
        if len(stream) < target_tokens:
            raise ValueError(
                f"Split {split} has only {len(stream)} tokens, below required {target_tokens}"
            )
        retained = stream[:target_tokens]
        retained_eos = [position for position in eos_positions if position < target_tokens]
        eos_valid = all(retained[position] == eos for position in retained_eos)
        if not eos_valid:
            raise ValueError(f"EOS boundary validation failed for {split}")
        _write_uint32(output_dir / f"{split}.bin", retained)
        _write_index(output_dir / f"{split}.idx", sequence_count, P5_SEQUENCE_LENGTH)
        split_memberships[split] = {item["document_id"] for item in memberships}
        split_reports[split] = {
            "sequence_count": sequence_count,
            "sequence_length": P5_SEQUENCE_LENGTH,
            "effective_tokens": target_tokens,
            "documents_consumed": len(memberships),
            "eos_tokens_retained": len(retained_eos),
            "eos_contract_valid": eos_valid,
            "dropped_tail_tokens": len(stream) - target_tokens,
            "final_incomplete_sequence_dropped": True,
            "cross_document_packing": True,
            "memberships": memberships,
        }
    overlaps = {
        "train_dev": sorted(split_memberships["train"] & split_memberships["dev"]),
        "train_heldout": sorted(split_memberships["train"] & split_memberships["heldout"]),
        "dev_heldout": sorted(split_memberships["dev"] & split_memberships["heldout"]),
    }
    manifest = {
        "schema_version": 1,
        "stage": "P5-Probe-corpus",
        "sequence_length": P5_SEQUENCE_LENGTH,
        "model_max_position_embeddings_modified": False,
        "long_term_context_requirement": {"required": 524_288, "stretch": 1_048_576},
        "document_split_first": True,
        "packing_second": True,
        "eos_between_documents": True,
        "token_dtype": "uint32_le",
        "index_dtype": "uint64_le",
        "source": source_report,
        "downloads": download_report,
        "tokenizer": tokenizer_report,
        "splits": split_reports,
        "document_overlap": overlaps,
        "document_overlap_count": sum(len(value) for value in overlaps.values()),
        "probe_train_ranges": P5_PROBE_RANGES,
    }
    manifest["manifest_hash_without_self"] = canonical_json_hash(manifest)
    (output_dir / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialized_hashes(path: str | Path) -> dict[str, str]:
    path = Path(path)
    names = [
        "train.bin",
        "train.idx",
        "dev.bin",
        "dev.idx",
        "heldout.bin",
        "heldout.idx",
        "corpus-manifest.json",
    ]
    return {name: sha256_file(path / name) for name in names}


def write_sha256sums(path: str | Path, hashes: dict[str, str]) -> None:
    path = Path(path)
    lines = [f"{digest}  packed/{name}" for name, digest in sorted(hashes.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_corpus_twice(
    data_dir: str | Path,
    tokenizer: Any,
    tokenizer_dir: str | Path,
) -> dict[str, Any]:
    """Build twice, compare hashes, then freeze the first build atomically."""
    data_dir = Path(data_dir).resolve()
    final_dir = data_dir / "packed"
    sums_path = data_dir / "SHA256SUMS"
    download_report = verify_download_manifest(data_dir)
    documents, source_report = load_candidate_documents(data_dir, tokenizer)
    second_documents, second_source_report = load_candidate_documents(data_dir, tokenizer)
    if source_report != second_source_report or documents != second_documents:
        raise ValueError("P5 corpus candidate selection is not deterministic")
    tokenizer_report = tokenizer_identity(tokenizer, tokenizer_dir)
    with tempfile.TemporaryDirectory(prefix="p5-corpus-a-", dir=data_dir) as first_root, tempfile.TemporaryDirectory(
        prefix="p5-corpus-b-", dir=data_dir
    ) as second_root:
        first = Path(first_root) / "packed"
        second = Path(second_root) / "packed"
        first_manifest = _materialize_once(
            documents,
            tokenizer,
            tokenizer_report,
            first,
            source_report,
            download_report,
        )
        _materialize_once(
            second_documents,
            tokenizer,
            tokenizer_report,
            second,
            second_source_report,
            download_report,
        )
        first_hashes = materialized_hashes(first)
        second_hashes = materialized_hashes(second)
        stable = first_hashes == second_hashes
        if not stable:
            raise ValueError("P5 corpus re-materialization hashes are not stable")
        if final_dir.exists():
            existing_hashes = materialized_hashes(final_dir)
            if existing_hashes != first_hashes:
                raise FileExistsError(
                    "Existing packed corpus differs from deterministic materialization; move it aside explicitly"
                )
        else:
            os.replace(first, final_dir)
        write_sha256sums(sums_path, first_hashes)
    return {
        "manifest": first_manifest,
        "hashes": first_hashes,
        "second_materialization_hashes": second_hashes,
        "re_materialization_hash_stable": stable,
        "packed_dir": str(final_dir),
        "sha256sums_path": str(sums_path),
    }


def verify_frozen_corpus(data_dir: str | Path) -> dict[str, Any]:
    """Replay every corpus invariant and file hash without rematerializing."""
    data_dir = Path(data_dir).resolve()
    packed = data_dir / "packed"
    manifest = json.loads((packed / "corpus-manifest.json").read_text(encoding="utf-8"))
    self_hash = manifest.pop("manifest_hash_without_self")
    hashes = materialized_hashes(packed)
    sums = {}
    for line in (data_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, declared = line.split(maxsplit=1)
        sums[Path(declared.strip()).name] = digest
    range_values = list(P5_PROBE_RANGES.values())
    ranges_non_overlapping = all(
        left["end_sequence"] <= right["start_sequence"]
        for left, right in zip(range_values, range_values[1:], strict=False)
    )
    checks = {
        "manifest_hash": canonical_json_hash(manifest) == self_hash,
        "all_sha256_verified": hashes == sums,
        "download_sha256_verified": manifest["downloads"]["all_sha256_verified"] is True,
        "document_overlap_zero": manifest["document_overlap_count"] == 0,
        "selected_documents_unique": all(
            manifest["splits"][split]["documents_consumed"]
            == len(
                {
                    item["document_id"]
                    for item in manifest["splits"][split]["memberships"]
                }
            )
            for split in P5_SPLIT_SEQUENCES
        )
        and sum(
            manifest["splits"][split]["documents_consumed"] for split in P5_SPLIT_SEQUENCES
        )
        == len(
            {
                item["text_sha256"]
                for split in P5_SPLIT_SEQUENCES
                for item in manifest["splits"][split]["memberships"]
            }
        ),
        "sequence_length_2048": manifest["sequence_length"] == P5_SEQUENCE_LENGTH,
        "split_counts_exact": all(
            manifest["splits"][split]["sequence_count"] == count
            and manifest["splits"][split]["effective_tokens"] == count * P5_SEQUENCE_LENGTH
            for split, count in P5_SPLIT_SEQUENCES.items()
        ),
        "membership_deterministic": manifest["document_split_first"] is True,
        "tokenizer_identity_frozen": len(manifest["tokenizer"]["identity_hash"]) == 64,
        "eos_contract_valid": all(
            manifest["splits"][split]["eos_contract_valid"] is True for split in P5_SPLIT_SEQUENCES
        ),
        "probe_ranges_non_overlapping": ranges_non_overlapping
        and range_values[0]["start_sequence"] == 0
        and range_values[-1]["end_sequence"] == P5_SPLIT_SEQUENCES["train"],
        "model_max_context_unchanged": manifest["model_max_position_embeddings_modified"] is False,
    }
    return {
        "stage": "P5-Probe-corpus",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hashes": hashes,
        "manifest_hash": self_hash,
        "probe_train_ranges": P5_PROBE_RANGES,
        "p5_probe_corpus_frozen": all(checks.values()),
        "p5_3_probe_training_allowed": all(checks.values()),
        "result_marker": f"P5_PROBE_CORPUS_MATERIALIZATION={'PASS' if all(checks.values()) else 'FAIL'}",
    }


class PackedTokenDataset:
    """Read one frozen fixed-length split through a read-only memory map."""

    def __init__(self, packed_dir: str | Path, split: str):
        if split not in P5_SPLIT_SEQUENCES:
            raise ValueError(f"Unknown P5 packed split: {split}")
        self.split = split
        self.sequence_length = P5_SEQUENCE_LENGTH
        packed_dir = Path(packed_dir)
        self.tokens = np.memmap(packed_dir / f"{split}.bin", mode="r", dtype="<u4")
        self.offsets = np.memmap(packed_dir / f"{split}.idx", mode="r", dtype="<u8")
        if len(self.offsets) != P5_SPLIT_SEQUENCES[split] + 1:
            raise ValueError(f"Invalid {split}.idx length")
        if int(self.offsets[-1]) != len(self.tokens):
            raise ValueError(f"Invalid {split} final token offset")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        if end - start != self.sequence_length:
            raise ValueError(f"Packed sequence {self.split}:{index} is not length {self.sequence_length}")
        return np.asarray(self.tokens[start:end], dtype=np.int64)


__all__ = [
    "P5_PROBE_RANGES",
    "P5_SEQUENCE_LENGTH",
    "P5_SPLIT_SEQUENCES",
    "PackedTokenDataset",
    "load_candidate_documents",
    "materialize_corpus_twice",
    "sha256_file",
    "verify_download_manifest",
    "verify_frozen_corpus",
]
