#!/usr/bin/env python3
"""Freeze source-only inputs for all available E-VQA-GLDv2 train rows.

The large encyclopedic KB stays compressed.  This script makes one streaming
pass through it and persists only the Wikipedia pages referenced by the local
1,891-row training subset.  QA text, answers, and evidence are deliberately
excluded from the resulting extraction manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit
import zipfile

import ijson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, image_info


DATA_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
SUBSET = "E-VQA-GLDv2-1898-61-seed0"
SUBSET_ROOT = DATA_ROOT / f"datasets_mm/E-VQA/subsets/{SUBSET}"
QA_TRAIN = SUBSET_ROOT / "qa_train.csv"
KB_ZIP = DATA_ROOT / "datasets_mm/E-VQA/raw/kb/encyclopedic_kb_wiki.zip"
OUTPUT_ROOT = DATA_ROOT / "expr_mm/evqa_api_strict_max_extraction_full1891_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    host = parsed.netloc.casefold().removeprefix("www.")
    path = unquote(parsed.path).rstrip("/") or "/"
    return f"{host}{path}".casefold()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_file = OUTPUT_ROOT / "source_manifest.json"
    if source_file.exists():
        existing = json.loads(source_file.read_text())
        if existing.get("status") == "complete":
            print(json.dumps({
                "status": "complete",
                "documents": existing["document_count"],
                "source_manifest": str(source_file),
            }, ensure_ascii=False))
            return

    subset_manifest = json.loads((SUBSET_ROOT / "manifest.json").read_text())
    images = {item["image_id"]: item for item in subset_manifest["copied_images"]}
    rows = list(csv.DictReader(QA_TRAIN.open(newline="")))
    if len(rows) != 1891:
        raise RuntimeError(f"expected 1,891 available train rows, found {len(rows)}")

    requested: dict[str, dict] = {}
    ordered_keys: list[str] = []
    for row_position, row in enumerate(rows):
        url = row["wikipedia_url"].split("|")[0].strip()
        key = normalize_url(url)
        image_id = row["dataset_image_ids"].split(",")[0].strip()
        image = images.get(image_id)
        if image is None:
            raise RuntimeError(f"subset manifest lacks image {image_id}")
        image_path = DATA_ROOT / image["subset_path"]
        if not image_info(image_path):
            raise RuntimeError(f"subset image cannot be decoded: {image_id}")
        if key not in requested:
            requested[key] = {
                "title": row["wikipedia_title"].strip(),
                "wikipedia_url": url,
                "image_id": image_id,
                "image_path": str(image_path),
                "image_ids": [],
                "image_paths": [],
                "row_positions": [],
            }
            ordered_keys.append(key)
        if image_id not in requested[key]["image_ids"]:
            requested[key]["image_ids"].append(image_id)
            requested[key]["image_paths"].append(str(image_path))
        requested[key]["row_positions"].append(row_position)

    selected_pages: dict[str, dict] = {}
    selected_raw_urls: dict[str, str] = {}
    with zipfile.ZipFile(KB_ZIP) as archive:
        names = archive.namelist()
        if names != ["encyclopedic_kb_wiki.json"]:
            raise RuntimeError(f"unexpected KB archive members: {names!r}")
        with archive.open(names[0]) as stream:
            for raw_url, page in ijson.kvitems(stream, ""):
                key = normalize_url(raw_url)
                if key in requested:
                    # The upstream KB contains a few HTTP/HTTPS aliases.  Keep
                    # the exact QA URL when present; otherwise the first alias
                    # is deterministic and has the same normalized identity.
                    exact = raw_url.rstrip("/") == requested[key]["wikipedia_url"].rstrip("/")
                    previous_exact = selected_raw_urls.get(key, "").rstrip("/") == \
                        requested[key]["wikipedia_url"].rstrip("/")
                    if key not in selected_pages or (exact and not previous_exact):
                        selected_pages[key] = page
                        selected_raw_urls[key] = raw_url

    missing = [requested[key]["wikipedia_url"] for key in ordered_keys
               if key not in selected_pages]
    if missing:
        raise RuntimeError(f"full KB lacks {len(missing)} requested Wikipedia pages: {missing[:10]}")

    documents = []
    for key in ordered_keys:
        identity = requested[key]
        page = selected_pages[key]
        section_texts = page.get("section_texts") or []
        section_titles = page.get("section_titles") or []
        if len(section_texts) != len(section_titles):
            raise RuntimeError(f"section arrays are misaligned for {identity['wikipedia_url']}")
        section_id = next((i for i, value in enumerate(section_texts)
                           if isinstance(value, str) and value.strip()), None)
        if section_id is None:
            raise RuntimeError(f"Wikipedia page has no non-empty section: {identity['wikipedia_url']}")
        title = identity["title"]
        section_title = str(section_titles[section_id]).strip()
        passage = str(section_texts[section_id]).strip()
        contents = f'"{title}"\nSection: {section_title}\n{passage}'
        document_id = "wiki::" + hashlib.sha256(
            f"{identity['wikipedia_url']}\0{section_id}".encode()
        ).hexdigest()[:20]
        documents.append({
            "document_id": document_id,
            "title": title,
            "wikipedia_url": identity["wikipedia_url"],
            "section_id": section_id,
            "section_title": section_title,
            "image_id": identity["image_id"],
            "image_path": identity["image_path"],
            "image_ids": identity["image_ids"],
            "image_paths": identity["image_paths"],
            "contents": contents,
            "contents_sha256": text_sha256(contents),
        })

    if len(documents) != 1237:
        raise RuntimeError(f"expected 1,237 unique train articles, found {len(documents)}")
    document_id_by_key = {
        normalize_url(document["wikipedia_url"]): document["document_id"]
        for document in documents
    }
    row_associations = []
    for row_position, row in enumerate(rows):
        url = row["wikipedia_url"].split("|")[0].strip()
        image_id = row["dataset_image_ids"].split(",")[0].strip()
        row_associations.append({
            "row_position": row_position,
            "document_id": document_id_by_key[normalize_url(url)],
            "image_id": image_id,
        })
    manifest = {
        "version": "evqa-full1891-train1237-first-section-v1",
        "status": "complete",
        "dataset": "E-VQA",
        "subset": SUBSET,
        "selection_policy": (
            "unique training-article URLs in CSV row order; complete first non-empty section; "
            "questions, answers, question types, and evidence labels are neither copied nor transmitted"
        ),
        "source_archive": str(KB_ZIP),
        "source_archive_sha256": sha256(KB_ZIP),
        "qa_train_sha256": sha256(QA_TRAIN),
        "available_train_row_count": len(rows),
        "document_count": len(documents),
        "unique_image_count": len({item["image_id"] for item in row_associations}),
        "row_association_count": len(row_associations),
        "row_associations": row_associations,
        "documents": documents,
    }
    atomic_json(source_file, manifest)
    print(json.dumps({
        "status": "complete", "train_rows": len(rows),
        "documents": len(documents), "source_manifest": str(source_file),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
