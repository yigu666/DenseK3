from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from densek3_core.recovery import p5_corpus


class TinyTokenizer:
    eos_token_id = 99
    special_tokens_map = {"eos_token": "<eos>"}

    def __len__(self) -> int:
        return 100

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [(ord(character) % 90) + 1 for character in text]


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_corpus_is_split_first_exact_and_reproducible(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p5_corpus, "P5_SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(p5_corpus, "P5_SPLIT_SEQUENCES", {"train": 2, "dev": 1, "heldout": 1})
    monkeypatch.setattr(
        p5_corpus,
        "P5_PROBE_RANGES",
        {
            "A": {"start_sequence": 0, "end_sequence": 1, "effective_tokens": 4},
            "B": {"start_sequence": 1, "end_sequence": 2, "effective_tokens": 4},
            "C": {"start_sequence": 2, "end_sequence": 2, "effective_tokens": 0},
        },
    )
    asset = tmp_path / "asset.txt"
    asset.write_text("frozen", encoding="utf-8")
    (tmp_path / "download-manifest.json").write_text(
        json.dumps({"files": [{"path": asset.name, "sha256": sha256(asset)}]}),
        encoding="utf-8",
    )
    records = []
    for index, split in enumerate(("train", "train", "dev", "heldout")):
        text = f"document-{index}"
        records.append(
            {
                "document_id": f"d{index}",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "split": split,
                "doc_rank": f"{index + 1:064x}",
            }
        )
    (tmp_path / "candidate-documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    result = p5_corpus.materialize_corpus_twice(tmp_path, TinyTokenizer(), tokenizer_dir)
    gate = p5_corpus.verify_frozen_corpus(tmp_path)
    assert result["re_materialization_hash_stable"] is True
    assert gate["status"] == "PASS"
    assert all(gate["checks"].values())
    assert gate["checks"]["selected_documents_unique"] is True
    assert "re_materialization_hash_stable" not in gate["checks"]
    dataset = p5_corpus.PackedTokenDataset(tmp_path / "packed", "train")
    assert len(dataset) == 2
    assert dataset[0].shape == (4,)
    assert dataset[0].dtype == np.int64
    assert dataset.tokens.dtype == np.dtype("<u4")
    assert (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").count("packed/") == 7


def test_real_parquet_candidate_schema_names_are_resolved(tmp_path, monkeypatch) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    monkeypatch.setattr(p5_corpus, "P5_SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(p5_corpus, "P5_SPLIT_SEQUENCES", {"train": 1, "dev": 1, "heldout": 1})
    raw = tmp_path / "raw/sample/10BT"
    raw.mkdir(parents=True)
    texts = ["abcdef", "ghijkl", "mnopqr"]
    parquet.write_table(pyarrow.table({"text": texts}), raw / "012_00000.parquet", row_group_size=1)
    records = []
    for index, (text, split) in enumerate(zip(texts, ("train", "dev", "heldout"), strict=True)):
        records.append(
            {
                "source_path": "sample/10BT/012_00000.parquet",
                "source_row_index": index,
                "source_document_id": f"doc-{index}",
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "qwen_token_count_including_eos": len(text) + 1,
                "split": split,
                "doc_rank": f"{index + 1:064x}",
            }
        )
    (tmp_path / "candidate-documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    documents, report = p5_corpus.load_candidate_documents(tmp_path, TinyTokenizer())
    assert [document.document_id for document in documents] == ["doc-0", "doc-1", "doc-2"]
    assert [document.text for document in documents] == texts
    assert report["schema_counts"] == {"embedded_text": 0, "parquet_reference": 3}
    assert all(document.source["source_row_index"] in {0, 1, 2} for document in documents)


def test_duplicates_are_globally_removed_before_split_selection(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p5_corpus, "P5_SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(p5_corpus, "P5_SPLIT_SEQUENCES", {"train": 2, "dev": 1, "heldout": 1})
    records = []
    rows = (
        ("train-first", "shared", "train", 1),
        ("dev-duplicate", "shared", "dev", 2),
        ("train-second", "unique-t", "train", 3),
        ("train-refill", "refill", "train", 4),
        ("dev-refill", "unique-d", "dev", 5),
        ("heldout", "unique-h", "heldout", 6),
    )
    for document_id, text, split, rank in rows:
        records.append(
            {
                "document_id": document_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "split": split,
                "doc_rank": f"{rank:064x}",
            }
        )
    (tmp_path / "candidate-documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    documents, report = p5_corpus.load_candidate_documents(tmp_path, TinyTokenizer())
    assert "dev-duplicate" not in {document.document_id for document in documents}
    assert report["duplicate_documents_excluded"] == 1
    assert report["duplicate_text_occurrences"] == 1
    assert len({document.document_id for document in documents}) == len(documents)
    assert len({document.text_sha256 for document in documents}) == len(documents)
    assert sum(document.expected_token_count_including_eos for document in documents if document.split == "dev") >= 4
