#!/usr/bin/env python3
"""Recover exact missing GLDv2 files from official moves / matching wiki SHA-1s.

Never substitute a similar landmark photograph or download a large original.
The existing downloader enforces direct networking, 8s pacing, Pillow validation,
1280px width, atomic no-clobber publication and shared-server safety guards.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
from urllib.parse import unquote, urlencode, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import download_gldv2_thumbnails as thumbnails
from evograph_mm.kb.gldv2_subset import (
    FORMAL_NAME, atomic_json, build_candidate_subset, prepare_plans,
)

WIKI_HOSTS = (
    "uk.wikipedia.org", "ru.wikipedia.org", "id.wikipedia.org",
    "en.wikipedia.org", "fr.wikipedia.org", "it.wikipedia.org", "lt.wikipedia.org",
)
HASH_ROUTES = {
    "d257f08beed642e3": ("ru.wikipedia.org", "en.wikipedia.org"),
    "17d641190089c6fb": ("fr.wikipedia.org", "en.wikipedia.org"),
    "d6775d2abe1c09d2": ("it.wikipedia.org", "en.wikipedia.org"),
    "bfb23baddcd34014": ("id.wikipedia.org", "en.wikipedia.org"),
    "2865f4e6a74bfc6f": ("id.wikipedia.org", "en.wikipedia.org"),
    "25a7ef7ec26d2c03": ("uk.wikipedia.org", "ru.wikipedia.org"),
}


def sha1_hex(value: str) -> str:
    """MediaWiki upload logs use base36; imageinfo may expose hexadecimal."""
    value = value.lower()
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    if not re.fullmatch(r"[0-9a-z]{1,31}", value):
        raise ValueError("invalid MediaWiki SHA-1")
    number = int(value, 36)
    if number >= 2**160:
        raise ValueError("MediaWiki SHA-1 exceeds 160 bits")
    return f"{number:040x}"


def identity_evidence(history: dict) -> dict:
    """Ignore deleted redirects and older unrelated files reusing the filename."""
    events = history["events"]
    moves = [e for e in events if e.get("type") == "move"
             and e.get("action") == "move" and e.get("params", {}).get("target_ns") == 6]
    deletes = [e for e in events if e.get("type") == "delete"
               and e.get("action") == "delete"]
    move = moves[0] if moves else None
    original_page = (move or (deletes[0] if deletes else {})).get("logpage")
    hashes = set()
    for e in events:
        if e.get("type") != "upload" or e.get("logpage") != original_page:
            continue
        value = e.get("params", {}).get("img_sha1")
        if value:
            hashes.add(sha1_hex(value))
    return {"expected_sha1s": sorted(hashes), "original_logpage": original_page,
            "move_target": move.get("params", {}).get("target_title") if move else None,
            "move_logid": move.get("logid") if move else None,
            "deletion_events": deletes}


def verified_sources(pages: dict, evidence: dict, host: str) -> dict:
    """Only exact historical original-file SHA-1 matches may be recovered."""
    found = {}
    for page in pages.values():
        for info in page.get("imageinfo", []):
            if not info.get("sha1"):
                continue
            observed = sha1_hex(info["sha1"])
            for image_id, proof in evidence.items():
                if observed not in proof["expected_sha1s"] or image_id in found:
                    continue
                url = info.get("thumburl")
                # A naturally small original is acceptable; never fetch a large one.
                if not url and 0 < info.get("width", 0) <= thumbnails.MAX_IMAGE_WIDTH:
                    url = info.get("url")
                if not url:
                    continue
                if (urlsplit(url).scheme != "https" or urlsplit(url).hostname not in
                        {"upload.wikimedia.org", "thumb.wikimedia.org"}):
                    raise thumbnails.SafeStop("unapproved_verified_image_host")
                found[image_id] = {"api_host": host, "title": page["title"],
                                   "sha1": observed, "thumbnail_url": url,
                                   "source_url": info.get("url"),
                                   "description_url": info.get("descriptionurl"),
                                   "identity_evidence": proof}
    return found


class VerifiedDownloader(thumbnails.Downloader):
    def __init__(self, root, logs):
        super().__init__(root, logs, 8)
        self.verified = {}

    def resolve(self, candidate):
        proof = self.verified.get(candidate["image_id"])
        if not proof:
            raise thumbnails.SafeStop("unverified_recovery_source")
        return proof["thumbnail_url"]


def api_query(client, host, params, metadata_route="proxy"):
    url = f"https://{host}/w/api.php?" + urlencode(params)
    if host == "commons.wikimedia.org":
        response = client.get(url, f"metadata:{host}")
        with response:
            data = response.json()
    else:
        # Only metadata uses this isolated explicit route; never inherit env keys
        # or leave proxies on the direct image-download session.
        direct_session = client.session
        with thumbnails.requests.Session() as metadata_session:
            metadata_session.trust_env = False
            metadata_session.headers.update(direct_session.headers)
            if metadata_route == "proxy":
                metadata_session.proxies.update({
                    "http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})
            client.emit("metadata_route", host=host, route=metadata_route)
            client.session = metadata_session
            try:
                response = client.get(url, f"metadata:{host}", connection_attempts=1)
                with response:
                    data = response.json()
            finally:
                client.session = direct_session
    if data.get("error"):
        raise thumbnails.SafeStop("MediaWiki_API_error:" + data["error"].get("code", "unknown"))
    query_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    atomic_json(client.logs / "gldv2_recovery_metadata" / f"{host}_{query_hash}.json",
                {"params": params, "response": data})
    return data


def image_query(client, host, titles, metadata_route="proxy"):
    params = {"action": "query", "format": "json", "prop": "imageinfo",
              "titles": "|".join(dict.fromkeys(titles)), "redirects": 1,
              "iiprop": "url|sha1|size|canonicaltitle|timestamp", "iiurlwidth": 1280}
    data = api_query(client, host, params, metadata_route)
    return data.get("query", {}).get("pages", {})


def files_with_sha1(client, host, digest, metadata_route="proxy"):
    # SHA-indexed lookup also finds cross-project transfers with translated names.
    data = api_query(client, host, {"action": "query", "format": "json",
                     "list": "allimages", "aisha1": digest, "ailimit": 10,
                     "aiprop": "sha1|canonicaltitle"}, metadata_route)
    return [i.get("title") or "File:" + i["name"]
            for i in data.get("query", {}).get("allimages", [])
            if i.get("sha1") and sha1_hex(i["sha1"]) == digest]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="download verified files and rebuild")
    parser.add_argument("--metadata-route", choices=("proxy", "direct"), default="proxy")
    parser.add_argument("--hash-search-only", action="store_true",
                        help="bounded official SHA-1 lookup regardless of translated filename")
    parser.add_argument("--hash-scope", choices=("targeted", "commons", "all"),
                        default="targeted", help="which official file indexes to search")
    args = parser.parse_args(argv)
    root = thumbnails.DEFAULT_ROOT
    logs = root / "logs"
    plan = prepare_plans(root)["formal"]
    prior_path = logs / f"{FORMAL_NAME}_downloads.json"
    prior = json.loads(prior_path.read_text())
    history = json.loads((logs / "gldv2_missing_file_history.json").read_text())
    if not history.get("complete"):
        parser.error("complete the official file-history lookup first")
    missing = {i for i, r in prior["results"].items() if r.get("status") != "valid"}
    evidence = {i: identity_evidence(history["results"][i]) for i in missing}
    candidates = {c["image_id"]: c for c in plan["candidates"] if c["image_id"] in missing}
    report_path = logs / "gldv2_verified_recovery.json"
    report = {"started_at": thumbnails.now(), "pid": os.getpid(), "complete": False,
              "missing_count_before": len(missing), "verified_sources": {},
              "recovered": {}, "evidence": evidence, "queried_hosts": [],
              "metadata_route": args.metadata_route,
              "hash_scope": args.hash_scope,
              "previously_recovered": [i for i, r in prior["results"].items()
                                       if r.get("identity_proof") and r.get("status") == "valid"]}
    with (logs / ".gldv2_download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Only official metadata APIs are added, in this isolated recovery process.
        thumbnails.ALLOWED_HOSTS.update(WIKI_HOSTS)
        client = VerifiedDownloader(root, logs)
        client.progress_path = logs / "gldv2_recovery_progress.json"
        client.events_path = logs / "gldv2_recovery_events.jsonl"
        client.state.update(phase="verified_missing_file_recovery", results={},
                            counts={"missing_before": len(missing), "recovered": 0})
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(client, "interrupted", True))

        def checkpoint():
            atomic_json(report_path, report)
            client.state["counts"]["recovered"] = len(report["recovered"])
            client.checkpoint()

        def accept(pages, host):
            matches = verified_sources(pages, evidence, host)
            for image_id, source in matches.items():
                if image_id in report["recovered"]:
                    continue
                report["verified_sources"][image_id] = source
                client.verified[image_id] = source
                if args.apply:
                    result = client.download(candidates[image_id])
                    result["identity_proof"] = source
                    client.state["results"][image_id] = result
                    if result["status"] == "valid":
                        report["recovered"][image_id] = result
                        prior["results"][image_id] = result
                        prior.update(counts=thumbnails.summarize(plan, prior["results"]),
                                     updated_at=thumbnails.now())
                        atomic_json(prior_path, prior)
                    client.emit("verified_recovery_result", image_id=image_id,
                                status=result["status"], source_host=host)
                checkpoint()

        try:
            if args.hash_search_only:
                report["hash_search"] = []
                for image_id, proof in evidence.items():
                    hosts = HASH_ROUTES.get(image_id, ())
                    if args.hash_scope == "commons":
                        hosts = ("commons.wikimedia.org",)
                    elif args.hash_scope == "all":
                        hosts = ("commons.wikimedia.org",) + hosts
                    for host in hosts:
                        for digest in proof["expected_sha1s"]:
                            lookup = {"image_id": image_id, "host": host, "sha1": digest}
                            try:
                                titles_found = files_with_sha1(client, host, digest, args.metadata_route)
                                lookup["titles"] = titles_found
                                if titles_found:
                                    accept(image_query(client, host, titles_found, args.metadata_route), host)
                            except (thumbnails.HTTPFailure, thumbnails.requests.RequestException) as error:
                                lookup["error"] = type(error).__name__
                                if isinstance(error, thumbnails.HTTPFailure):
                                    lookup["http_status"] = error.status
                            report["hash_search"].append(lookup)
                            checkpoint()
                        if image_id in report["recovered"]:
                            break
            moved = [p["move_target"] for p in evidence.values() if p["move_target"]]
            if moved:
                accept(image_query(client, "commons.wikimedia.org", moved), "commons.wikimedia.org")
                report["queried_hosts"].append("commons.wikimedia.org")
            titles = ["File:" + unquote(urlsplit(c["url"]).path.rsplit("/", 1)[-1])
                      for c in candidates.values()]
            for host in (() if args.hash_search_only else WIKI_HOSTS):
                if len(report["recovered"]) == len(missing):
                    break
                try:
                    accept(image_query(client, host, titles, args.metadata_route), host)
                except (thumbnails.HTTPFailure, thumbnails.requests.RequestException) as error:
                    report.setdefault("metadata_errors", {})[host] = str(type(error).__name__)
                    if isinstance(error, thumbnails.HTTPFailure):
                        report["metadata_errors"][host] += f":HTTP_{error.status}"
                report["queried_hosts"].append(host)
                checkpoint()
            if args.apply:
                rebuilt = build_candidate_subset(root, plan)
                rebuilt["downloads"] = prior
                atomic_json(logs / "evqa_gldv2_1898_61_summary.json", rebuilt)
                report["subset_summary"] = rebuilt["summary"]
                report["unresolved"] = rebuilt["missing"]
                client.state["formal_subset_complete"] = rebuilt["complete"]
            report.update(complete=True, finished_at=thumbnails.now())
            client.state["status"] = "complete"
        except (thumbnails.SafeStop, ValueError, OSError) as error:
            report.update(stop_reason=str(error), finished_at=thumbnails.now())
            client.state.update(status="stopped", stop_reason=str(error))
        finally:
            checkpoint()
            client.session.close()
    print(json.dumps({"complete": report["complete"], "recovered": list(report["recovered"]),
                      "subset_summary": report.get("subset_summary"),
                      "stop_reason": report.get("stop_reason")}, ensure_ascii=False))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
