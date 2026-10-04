"""Candidate-driven Encyclopedic VQA subsets, without resampling or deleting images."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

from PIL import Image

from .subset_echosight import SUBSET_MANIFEST_VERSION, _image_id_from_row, _sha256
from .validate_echosight import _subset_manifest_report

DATASET = "E-VQA"
FORMAL_NAME = "E-VQA-GLDv2-1898-61-seed0"
SMALL_NAME = "paper_64_16_seed0"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2, ensure_ascii=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def image_info(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            return {"width": image.width, "height": image.height,
                    "format": image.format, "bytes": path.stat().st_size}
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def publish_image(temp: Path, target: Path) -> None:
    """Linux atomic rename with RENAME_NOREPLACE: never replace an existing file."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temp), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(code, os.strerror(code), str(target))


def local_image(selected: Path, image_id: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = selected / f"{image_id}{suffix}"
        if image_info(path):
            return path
    return None


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qa_rows(raw: Path, manifests: list[dict]) -> tuple[dict, dict, dict]:
    wanted = {s: set() for s in ("train", "test")}
    for manifest in manifests:
        for c in manifest["candidates"]:
            if c.get("split") not in wanted or type(c.get("row_index")) is not int:
                raise ValueError("candidate has invalid split/row_index")
            wanted[c["split"]].add(c["row_index"])
    rows, fields, counts = {}, {}, {}
    for split in wanted:
        rows[split] = {}
        with (raw / f"qa_{split}.csv").open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields[split] = list(reader.fieldnames or [])
            count = 0
            for index, row in enumerate(reader):
                if index in wanted[split]:
                    rows[split][index] = dict(row)
                count = index + 1
            counts[split] = count
        if len(rows[split]) != len(wanted[split]):
            raise ValueError(f"candidate indices outside QA CSV: {split}")
    return rows, fields, counts


def prepare_plans(root: Path) -> dict:
    base = root / "datasets_mm" / DATASET
    raw = base / "raw"
    paths = [base / "subsets" / n / "selected_image_candidates.json"
             for n in (SMALL_NAME, "paper_5120_128_seed0")]
    small, original = [_json(p) for p in paths]
    for m in (small, original):
        if m.get("dataset") != DATASET or m.get("seed") != 0:
            raise ValueError("candidate dataset/seed mismatch")
    if (small.get("sample_train"), small.get("sample_test")) != (64, 16):
        raise ValueError("small candidate counts mismatch")
    if (original.get("sample_train"), original.get("sample_test")) != (5120, 128):
        raise ValueError("original candidate counts mismatch")
    rows, fields, counts = _qa_rows(raw, [small, original])
    rng = random.Random(0)
    formal = []
    for split, size in (("train", 5120), ("test", 128)):
        order = list(range(counts[split]))
        rng.shuffle(order)
        candidates = [c for c in original["candidates"] if c["split"] == split]
        # Check the entire buffered sequence, then slice BEFORE provider filtering.
        if [c["row_index"] for c in candidates] != order[:len(candidates)]:
            raise ValueError(f"original candidate list is not deterministic seed=0: {split}")
        selected = candidates[:size]
        if len(selected) != size:
            raise ValueError(f"insufficient original mixed candidates: {split}")
        for ordinal, candidate in enumerate(selected):
            if candidate.get("source", candidate.get("provider")) == "gldv2":
                formal.append({**candidate, "original_mixed_ordinal": ordinal})
    formal_counts = {s: sum(c["split"] == s for c in formal) for s in ("train", "test")}
    if formal_counts != {"train": 1898, "test": 61}:
        raise ValueError(f"fixed GLDv2 split differs from expected: {formal_counts}")
    small_candidates = small["candidates"]
    for split in ("train", "test"):
        buffered_gld = [c for c in original["candidates"]
                        if c["split"] == split and c["source"] == "gldv2"]
        small_split = [c for c in small_candidates if c["split"] == split]
        if small_split != buffered_gld[:len(small_split)]:
            raise ValueError(f"small candidates disagree with original deterministic order: {split}")
    metadata_ids = set()
    for c in small_candidates + formal:
        row = rows[c["split"]][c["row_index"]]
        field, image_id = _image_id_from_row(row)
        if (c.get("source", c.get("provider")) != "gldv2"
                or not re.fullmatch(r"[0-9a-fA-F]{16}", str(c.get("image_id")))
                or image_id != c["image_id"] or field != c["image_field"]
                or row.get("dataset_name") != "landmarks"
                or not row.get("question") or not row.get("answer")):
            raise ValueError(f"candidate cannot be reliably matched to QA: {c}")
        metadata_ids.add(image_id)
    metadata_path = raw / "images/google_landmarks_v2/metadata/train.csv"
    metadata = {}
    with metadata_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_id = row.get("id")
            if image_id in metadata_ids:
                if image_id in metadata and metadata[image_id] != row["url"]:
                    raise ValueError(f"ambiguous GLDv2 metadata: {image_id}")
                metadata[image_id] = row["url"]
    if metadata_ids - metadata.keys():
        raise ValueError(f"missing GLDv2 metadata: {sorted(metadata_ids - metadata.keys())}")
    for url in metadata.values():
        if urlsplit(url).hostname not in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
            raise ValueError("unsupported metadata host; no unverified fallback downloads")

    def enrich(candidates: list) -> list:
        result = []
        for c in candidates:
            url = metadata[c["image_id"]]
            suffix = Path(unquote(urlsplit(url).path)).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                suffix = ".jpg"
            row = rows[c["split"]][c["row_index"]]
            digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            result.append({**c, "provider": "gldv2", "url": url,
                           "target_path": str(raw / "images/google_landmarks_v2/selected"
                                              / f"{c['image_id']}{suffix}"),
                           "qa_row": row, "qa_sha256": digest})
        return result

    common = {"dataset": DATASET, "seed": 0, "qa_fields": fields,
              "source_files": {s: str(raw / f"qa_{s}.csv") for s in fields}}
    return {
        "small": {**common, "name": SMALL_NAME, "requested": {"train": 64, "test": 16},
                  "allow_buffer": True, "candidates": enrich(small_candidates),
                  "source_candidates": str(paths[0]), "source_sha256": _sha256(paths[0])},
        "formal": {**common, "name": FORMAL_NAME, "requested": formal_counts,
                   "allow_buffer": False, "candidates": enrich(formal),
                   "source_candidates": str(paths[1]), "source_sha256": _sha256(paths[1]),
                   "original_sample_train": 5120, "original_sample_test": 128},
    }


def build_candidate_subset(root: Path, plan: dict) -> dict:
    """Build existing EchoSight manifest/CSV format from exact, verified QA rows."""
    subset_rel = Path("datasets_mm") / DATASET / "subsets" / plan["name"]
    subset = root / subset_rel
    if subset.resolve().parent != (root / "datasets_mm" / DATASET / "subsets").resolve():
        raise ValueError("unsafe subset output path")
    images = subset / "images"
    images.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": [], "test": []}
    copied, chosen, missing = [], [], []
    for c in plan["candidates"]:
        split = c["split"]
        if plan["allow_buffer"] and len(split_rows[split]) >= plan["requested"][split]:
            continue
        source = local_image(Path(c["target_path"]).parent, c["image_id"])
        if source is None:
            missing.append({k: c[k] for k in ("split", "row_index", "image_id", "url")})
            continue
        target = images / source.name
        if target.exists():
            if not image_info(target) or _sha256(target) != _sha256(source):
                raise ValueError(f"preserving existing different/invalid subset image: {target}")
        else:
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=images)
            os.close(fd)
            temp = Path(name)
            try:
                shutil.copyfile(source, temp)
                if not image_info(temp):
                    raise ValueError(f"image failed decoding before publishing: {source}")
                publish_image(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        split_rows[split].append(c["qa_row"])
        chosen.append(c)
        copied.append({"split": split, "row_index": c["row_index"],
                       "image_id": c["image_id"], "source_path": str(source),
                       "subset_path": str(subset_rel / "images" / target.name),
                       "sha256": _sha256(target)})
    for split, values in split_rows.items():
        path = subset / f"qa_{split}.csv"
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=subset)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=plan["qa_fields"][split])
                writer.writeheader()
                writer.writerows(values)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    counts = {s: len(v) for s, v in split_rows.items()}
    complete = counts == plan["requested"]
    blockers = [] if complete else [{"reason": "missing_valid_images", "missing": missing}]
    summary = {"status": "complete" if complete else "subset_incomplete",
               "complete_train": counts["train"], "complete_test": counts["test"],
               "requested_train": plan["requested"]["train"],
               "requested_test": plan["requested"]["test"]}
    manifest = {"manifest_version": SUBSET_MANIFEST_VERSION, "dataset": DATASET,
                "subset_root": str(subset_rel), "seed": plan["seed"],
                "sample_train": plan["requested"]["train"],
                "sample_test": plan["requested"]["test"],
                "source_files": plan["source_files"], "summary": summary,
                "copied_images": copied, "blockers": blockers,
                "candidate_provenance": {k: plan[k] for k in
                                         ("source_candidates", "source_sha256", "allow_buffer")},
                "selected_candidates": [{k: c[k] for k in
                                         ("split", "row_index", "image_id", "qa_sha256")}
                                        for c in chosen]}
    atomic_json(subset / "manifest.json", manifest)
    validation = _subset_manifest_report(root, DATASET, subset / "manifest.json",
                                        max_json_bytes=50_000_000)
    # Verify the written CSVs, ALL images and exact raw QA field correspondence.
    correspondence = True
    empty_evidence = 0
    for split in split_rows:
        with (subset / f"qa_{split}.csv").open(newline="", encoding="utf-8") as f:
            observed = list(csv.DictReader(f))
        correspondence &= observed == split_rows[split]
        empty_evidence += sum(not row.get("evidence", "").strip() for row in observed)
    all_images_valid = all(image_info(root / i["subset_path"]) for i in copied)
    report = {"subset": plan["name"], "summary": summary, "missing": missing,
              "repository_validation": validation,
              "qa_correspondence_valid": correspondence, "all_images_decodable": all_images_valid,
              "empty_evidence_preserved_from_raw": empty_evidence,
              "complete": complete and correspondence and all_images_valid
                          and validation["status"] == "complete"}
    atomic_json(subset / "validation_report.json", report)
    return report
