"""Provenance and checksum manifest generation for immutable inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .benchmark_loaders import benchmark_inventory


OFFICIAL_REPOSITORIES = {
    "fever": "https://github.com/sheffieldnlp/naacl2018-fever.git",
    "fever-scorer": "https://github.com/sheffieldnlp/fever-scorer.git",
    "hover": "https://github.com/hover-nlp/hover.git",
    "averitec": "https://github.com/MichSchli/AVeriTeC.git",
    "expertqa": "https://github.com/chaitanyamalaviya/ExpertQA.git",
}

OFFICIAL_DOWNLOADS = {
    "fever_train": "https://fever.ai/download/fever/train.jsonl",
    "fever_dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "fever_wikipedia_2017": "https://fever.ai/download/fever/wiki-pages.zip",
    "averitec_dev_knowledge_store": (
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/"
        "data_store/knowledge_store/dev_knowledge_store.zip"
    ),
    "wikidata_20170220": (
        "https://archive.org/download/wikibase-wikidatawiki-20170220/"
        "wikidata-20170220-all.json.bz2"
    ),
}

OFFICIAL_FILE_EXPECTATIONS = {
    "averitec_dev_knowledge_store": {
        "bytes": 11_537_899_362,
        "sha256": "021e258cd6fb5fe6d627a4667d663e95c184c966939c15124df9206142fc2212",
    },
    "wikidata_20170220": {
        "bytes": 7_166_563_790,
        "md5": "3e5df7fcf8b5b9c893f403a631c70537",
        "sha1": "ba880b53975224df7dcdf54fc9099e4882aaee74",
        "date": "2017-02-20",
        "license": "CC0-1.0",
    },
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    return file_digest(path, "sha256", chunk_size=chunk_size)


def file_digest(
    path: Path, algorithm: str, *, chunk_size: int = 1024 * 1024
) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_file_preflight(name: str, path: Path) -> None:
    """Reject absent or truncated immutable inputs before expensive hashing."""
    if not path.is_file():
        raise FileNotFoundError(path)
    expectation = OFFICIAL_FILE_EXPECTATIONS.get(name)
    if expectation and path.stat().st_size != expectation["bytes"]:
        raise ValueError(
            f"{name} has {path.stat().st_size} bytes; expected {expectation['bytes']}"
        )


def build_manifest(
    project_root: Path,
    include_wikipedia: bool = True,
    include_wikidata: bool = False,
) -> dict[str, Any]:
    external = project_root / ".external"
    raw = project_root / "data" / "raw"
    repositories = {}
    for name, url in OFFICIAL_REPOSITORIES.items():
        path = external / f"{name}-official"
        repositories[name] = {"url": url, "commit": git_commit(path)}
    files = {
        "fever_train": raw / "fever_train.jsonl",
        "fever_dev": raw / "fever_dev.jsonl",
        "hover_train": external / "hover-official/data/hover/hover_train_release_v1.1.json",
        "hover_dev": external / "hover-official/data/hover/hover_dev_release_v1.1.json",
        "hover_test": external / "hover-official/data/hover/hover_test_release_v1.1.json",
        "averitec_train": external / "averitec-official/data/train.json",
        "averitec_dev": external / "averitec-official/data/dev.json",
        "averitec_dev_knowledge_store": raw / "averitec_dev_knowledge_store.zip",
        "expertqa_train": external / "expertqa-official/data/lfqa/domain_lfqa_train.json",
        "expertqa_dev": external / "expertqa-official/data/lfqa/domain_lfqa_val.json",
        "expertqa_test": external / "expertqa-official/data/lfqa/domain_lfqa_test.json",
        "expertqa_annotated": external / "expertqa-official/data/r2_compiled_anon.jsonl",
        "expertqa_all_passage_evidence": external
        / "expertqa-official/data/r2_compiled_all_evidences_autoais_anon.jsonl",
    }
    if include_wikipedia:
        files["fever_wikipedia_2017"] = raw / "wiki-pages.zip"
    if include_wikidata:
        files["wikidata_20170220"] = raw / "wikidata-20170220-all.json.bz2"
    for name, path in files.items():
        validate_file_preflight(name, path)
    file_records = {}
    for name, path in files.items():
        expectation = OFFICIAL_FILE_EXPECTATIONS.get(name)
        digest = sha256(path)
        if expectation and expectation.get("sha256") and digest != expectation["sha256"]:
            raise ValueError(
                f"{name} checksum mismatch: {digest} != {expectation['sha256']}"
            )
        record = {
            "path": str(path.relative_to(project_root)),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "url": OFFICIAL_DOWNLOADS.get(name),
        }
        for algorithm in ("md5", "sha1"):
            expected = expectation.get(algorithm) if expectation else None
            if expected:
                observed = file_digest(path, algorithm)
                if observed != expected:
                    raise ValueError(
                        f"{name} {algorithm} mismatch: {observed} != {expected}"
                    )
                record[algorithm] = observed
        for field in ("date", "license"):
            if expectation and field in expectation:
                record[field] = expectation[field]
        file_records[name] = record
    return {
        "schema_version": 1,
        "repositories": repositories,
        "files": file_records,
        "inventory": benchmark_inventory(project_root),
    }


def write_manifest(
    project_root: Path,
    output: Path,
    include_wikipedia: bool = True,
    include_wikidata: bool = False,
) -> None:
    manifest = build_manifest(
        project_root,
        include_wikipedia=include_wikipedia,
        include_wikidata=include_wikidata,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
