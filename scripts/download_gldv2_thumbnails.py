#!/usr/bin/env python3
"""Direct, paced, resumable Wikimedia thumbnails for exact E-VQA GLDv2 candidates.

Run --stage probe first (five missing images), then --stage pipeline. No proxy,
GPU, credentials, full archives, or iNaturalist downloads are used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import (  # noqa: E402
    FORMAL_NAME, atomic_json, build_candidate_subset, image_info,
    local_image, prepare_plans, publish_image,
)

USER_AGENT = "EvoGraph-EVQA-Academic-Downloader/1.0"
MAX_IMAGE_WIDTH = 1280
DEFAULT_ROOT = Path("/home/data/dataset/wjz/EvoGraph-R1")
BACKOFF = (60, 300, 900)
ALLOWED_HOSTS = {"commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org"}


class SafeStop(Exception):
    pass


class HTTPFailure(Exception):
    def __init__(self, status: int, retry_after: str | None = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"HTTP {status}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    # Never log user info or query parameters, which could contain credentials.
    return urlunsplit((parts.scheme, parts.hostname or "", parts.path, "", ""))


def retry_after_seconds(value: str | None, current: float | None = None) -> float | None:
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0, date.timestamp() - (time.time() if current is None else current))
        except (ValueError, TypeError, OverflowError):
            return None


def imageinfo_url(source: str) -> str:
    filename = unquote(urlsplit(source).path.rsplit("/", 1)[-1])
    return "https://commons.wikimedia.org/w/api.php?" + urlencode({
        "action": "query", "format": "json", "prop": "imageinfo", "redirects": 1,
        "titles": "File:" + filename, "iiprop": "url|size", "iiurlwidth": MAX_IMAGE_WIDTH,
    })


def redirect_url(source: str) -> str:
    filename = unquote(urlsplit(source).path.rsplit("/", 1)[-1])
    return ("https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            + quote(filename) + f"?width={MAX_IMAGE_WIDTH}")


def limit_image_width(temp: Path, target: Path) -> None:
    """Cap new thumbnails at 1280px without upscaling smaller images."""
    with Image.open(temp) as image:
        image.load()
        if image.width > MAX_IMAGE_WIDTH:
            height = max(1, round(image.height * MAX_IMAGE_WIDTH / image.width))
            resized = image.resize((MAX_IMAGE_WIDTH, height), Image.Resampling.LANCZOS)
            if target.suffix.lower() in {".jpg", ".jpeg"}:
                resized.convert("RGB").save(temp, format="JPEG", quality=92)
            else:
                resized.save(temp, format="PNG")


class Downloader:
    def __init__(self, root: Path, logs: Path, interval: float = 8):
        if interval < 8:
            raise ValueError("request interval must be at least 8 seconds")
        self.root, self.logs, self.interval = root, logs, interval
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.last_request = 0.0
        self.connection_failure_since = None
        self.interrupted = False
        self.consecutive_429 = 0
        self.payloads = 0
        self.invalid_payloads = 0
        self.state = {"pid": os.getpid(), "started_at": now(), "status": "running",
                      "phase": "planning", "results": {}, "request_count": 0,
                      "max_image_width": MAX_IMAGE_WIDTH}
        self.progress_path = logs / "gldv2_download_progress.json"
        self.events_path = logs / "gldv2_download_events.jsonl"

    def emit(self, event: str, **values):
        record = {"time": now(), "event": event, **values}
        with self.events_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def checkpoint(self):
        self.state.update(updated_at=now(), consecutive_429=self.consecutive_429,
                          free_disk_bytes=shutil_disk_free(self.root))
        atomic_json(self.progress_path, self.state)

    def guard(self):
        if self.interrupted:
            raise SafeStop("termination_signal_received")
        if shutil_disk_free(self.root) < 100_000_000_000:
            raise SafeStop("free_disk_below_100GB")
        if self.consecutive_429 >= 20:
            raise SafeStop("20_consecutive_HTTP_429_responses")
        if (self.connection_failure_since is not None
                and time.monotonic() - self.connection_failure_since >= 1800):
            raise SafeStop("direct_connection_unavailable_over_30_minutes")
        if self.payloads >= 10 and self.invalid_payloads / self.payloads >= 0.2:
            raise SafeStop("image_decode_failure_rate_at_least_20_percent_in_10_payloads")

    def wait(self, seconds: float, reason: str):
        if seconds <= 0:
            return
        self.state["waiting"] = {"reason": reason, "seconds": seconds,
                                 "until_epoch": time.time() + seconds}
        self.checkpoint()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.guard()
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        self.state.pop("waiting", None)

    def get(self, url: str, image_id: str, stream=False, connection_attempts=4):
        """Every HTTP request and redirect is paced and has bounded timeouts."""
        if not 1 <= connection_attempts <= 4:
            raise ValueError("connection_attempts must be between 1 and 4")
        for redirect in range(5):
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in ALLOWED_HOSTS:
                raise SafeStop("unapproved_redirect_host_or_insecure_URL")
            for attempt in range(4):
                self.guard()
                self.wait(self.interval - (time.monotonic() - self.last_request), "request_spacing")
                self.last_request = time.monotonic()
                self.state["request_count"] += 1
                try:
                    response = self.session.get(url, timeout=(15, 60), stream=stream,
                                                allow_redirects=False)
                except requests.RequestException as error:
                    if self.connection_failure_since is None:
                        self.connection_failure_since = time.monotonic()
                    self.emit("connection_error", image_id=image_id, error=type(error).__name__,
                              host=urlsplit(url).hostname, attempt=attempt + 1)
                    self.checkpoint()
                    if attempt >= connection_attempts - 1:
                        raise
                    self.wait(BACKOFF[attempt], "direct_connection_retry")
                    continue
                self.connection_failure_since = None
                status = response.status_code
                self.consecutive_429 = self.consecutive_429 + 1 if status == 429 else 0
                self.emit("http", image_id=image_id, status=status, host=urlsplit(url).hostname,
                          attempt=attempt + 1)
                if status == 429:
                    retry_after = response.headers.get("Retry-After")
                    response.close()
                    self.guard()
                    delay = retry_after_seconds(retry_after)
                    # Persist the last cooldown even when deferring this image.
                    delay = max(self.interval, delay if delay is not None
                                else BACKOFF[min(attempt, 2)])
                    self.wait(delay, "HTTP_429_retry_after")
                    if attempt == 3:
                        raise HTTPFailure(status, retry_after)
                    continue
                if status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    response.close()
                    if not location:
                        raise HTTPFailure(status)
                    url = urljoin(url, location)
                    break
                if status != 200:
                    response.close()
                    raise HTTPFailure(status)
                return response
            else:
                raise HTTPFailure(429)
        raise SafeStop("too_many_thumbnail_redirects")

    def resolve(self, candidate: dict) -> str:
        source, image_id = candidate["url"], candidate["image_id"]
        response = self.get(imageinfo_url(source), image_id)
        try:
            data = response.json()
        except ValueError:
            self.emit("imageinfo_not_JSON", image_id=image_id)
            return redirect_url(source)
        finally:
            response.close()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return redirect_url(source)
        for page in pages.values():
            for info in page.get("imageinfo", []):
                if info.get("thumburl"):
                    return info["thumburl"]
                # Do not fall back to original large images.
                return redirect_url(source)
        raise HTTPFailure(404)

    def download(self, c: dict) -> dict:
        target = Path(c["target_path"])
        selected = self.root / "datasets_mm/E-VQA/raw/images/google_landmarks_v2/selected"
        if target.parent.resolve() != selected.resolve():
            raise SafeStop("image_target_outside_selected_directory")
        existing = local_image(selected, c["image_id"])
        if existing:
            return {"status": "valid", "reused": True, "target_path": str(existing),
                    "image": image_info(existing)}
        if target.exists():
            raise SafeStop(f"existing_invalid_image_requires_approval:{c['image_id']}")
        temp = None
        result = {"image_id": c["image_id"], "source_url": safe_url(c["url"]),
                  "target_path": str(target)}
        try:
            thumbnail = self.resolve(c)
            result["thumbnail_url"] = safe_url(thumbnail)
            response = self.get(thumbnail, c["image_id"], stream=True)
            result["http_status"] = response.status_code
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
            temp = Path(name)
            size = 0
            with response, os.fdopen(fd, "wb") as out:
                for chunk in response.iter_content(64 * 1024):
                    self.guard()
                    size += len(chunk)
                    if size > 25_000_000:
                        raise ValueError("thumbnail_payload_exceeds_25MB")
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            self.payloads += 1
            if not image_info(temp):
                self.invalid_payloads += 1
                raise ValueError("HTTP_200_payload_not_a_decodable_image")
            limit_image_width(temp, target)
            if not image_info(temp):
                raise ValueError("resized_image_not_decodable")
            try:
                publish_image(temp, target)
            except FileExistsError:
                if not image_info(target):
                    raise SafeStop("concurrent_invalid_image_preserved")
            result.update(status="valid", reused=False, image=image_info(target))
        except HTTPFailure as error:
            result.update(status="failed", http_status=error.status,
                          retry_after=error.retry_after,
                          reason="Wikimedia_file_missing" if error.status == 404
                          else f"HTTP_{error.status}")
        except requests.RequestException as error:
            result.update(status="failed", reason=f"direct_connection_{type(error).__name__}")
        except (OSError, ValueError) as error:
            result.update(status="failed", reason=str(error)[:200])
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        result["updated_at"] = now()
        # New sidecar preserves the old selected_download sidecar for provenance.
        atomic_json(target.with_name(target.name + ".thumbnail_download.json"), result)
        self.guard()
        return result

    def run(self, plan: dict, limit: int | None = None) -> dict:
        by_id = {}
        for c in plan["candidates"]:
            by_id.setdefault(c["image_id"], c)
        candidates = list(by_id.values())
        prior_path = self.logs / f"{plan['name']}_downloads.json"
        prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
        results = prior.get("results", {})
        # Count all already verified images before selecting five missing URLs.
        # Stale cached successes are never trusted without decoding the image.
        for c in candidates:
            path = local_image(Path(c["target_path"]).parent, c["image_id"])
            if path:
                results[c["image_id"]] = {"status": "valid", "reused": True,
                                          "target_path": str(path), "image": image_info(path)}
            elif results.get(c["image_id"], {}).get("status") == "valid":
                results.pop(c["image_id"])
        self.state.update(phase=plan["name"], sample_counts=plan["requested"],
                          total_unique_images=len(candidates), results=results)
        self.state["counts"] = summarize(plan, results)
        self.checkpoint()
        if limit is not None:
            candidates = [c for c in candidates
                          if not local_image(Path(c["target_path"]).parent, c["image_id"])][:limit]
        for c in candidates:
            self.guard()
            self.state["current_image_id"] = c["image_id"]
            old = results.get(c["image_id"], {})
            if (old.get("http_status") == 404
                    and not local_image(Path(c["target_path"]).parent, c["image_id"])):
                continue
            results[c["image_id"]] = self.download(c)
            self.state["results"] = results
            counts = summarize(plan, results)
            self.state["counts"] = counts
            report = {"plan": plan["name"], "counts": counts, "results": results,
                      "updated_at": now()}
            atomic_json(prior_path, report)
            self.checkpoint()
            self.emit("image_result", image_id=c["image_id"],
                      status=results[c["image_id"]]["status"], counts=counts)
        report = {"plan": plan["name"], "counts": summarize(plan, results),
                  "results": results, "updated_at": now()}
        atomic_json(prior_path, report)
        return report


def shutil_disk_free(root: Path) -> int:
    return shutil.disk_usage(root).free


def summarize(plan: dict, results: dict) -> dict:
    ids = {c["image_id"] for c in plan["candidates"]}
    valid = sum(results.get(i, {}).get("status") == "valid" for i in ids)
    failed = sum(results.get(i, {}).get("status") == "failed" for i in ids)
    return {"total": len(ids), "successful": valid, "failed": failed,
            "pending": len(ids) - valid - failed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("inspect", "probe", "pipeline", "launch"), required=True)
    parser.add_argument("--interval", type=float, default=8)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if root != DEFAULT_ROOT:
        parser.error("CLI writes are restricted to the configured wjz dataset root")
    if args.stage == "launch":
        logs = root / "logs"
        probe_path = logs / "gldv2_probe_report.json"
        if not probe_path.is_file() or not json.loads(probe_path.read_text()).get("passed"):
            parser.error("complete the five-image probe before launching background work")
        if not shutil.which("nohup") or not shutil.which("setsid"):
            parser.error("nohup and setsid are required for this server's background launcher")
        with (logs / ".gldv2_download.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another downloader is already active")
            with (logs / "gldv2_download.log").open("a") as log:
                process = subprocess.Popen(
                    ["nohup", "setsid", sys.executable, str(Path(__file__).resolve()),
                     "--stage", "pipeline", "--interval", str(args.interval)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    close_fds=True,
                )
            atomic_json(logs / "gldv2_background_launch.json",
                        {"pid": process.pid, "launched_at": now(),
                         "method": "nohup setsid", "log": str(logs / "gldv2_download.log")})
            (logs / "gldv2_download.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "log": str(logs / "gldv2_download.log")}))
        return 0
    plans = prepare_plans(root)
    if args.stage == "inspect":
        print(json.dumps({k: {"name": p["name"], "rows": len(p["candidates"]),
                              "splits": p["requested"],
                              "unique_images": len({c["image_id"] for c in p["candidates"]})}
                          for k, p in plans.items()}, indent=2))
        return 0
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / ".gldv2_download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another thumbnail downloader is already active", file=sys.stderr)
            return 2
        downloader = Downloader(root, logs, args.interval)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(downloader, "interrupted", True))
        atomic_json(logs / "gldv2_download_plan.json", plans)
        try:
            if args.stage == "probe":
                report = downloader.run(plans["small"], limit=5)
                successes = [i for i, r in report["results"].items()
                             if r.get("status") == "valid" and not r.get("reused")]
                probe = {**report, "successful_thumbnail_ids": successes,
                         "passed": bool(successes)}
                atomic_json(logs / "gldv2_probe_report.json", probe)
                downloader.state.update(status="probe_passed" if successes else "probe_failed")
                return 0 if successes else 1
            probe_path = logs / "gldv2_probe_report.json"
            probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
            if not probe.get("passed"):
                raise SafeStop("five_image_probe_not_successful")
            small_download = downloader.run(plans["small"])
            small = build_candidate_subset(root, plans["small"])
            small["downloads"] = small_download
            atomic_json(logs / "evqa_gldv2_64_16_summary.json", small)
            if not small["complete"]:
                raise SafeStop("small_subset_incomplete_formal_download_not_started")
            # Stage two starts only AFTER repository validation of stage one.
            formal_dir = root / "datasets_mm/E-VQA/subsets" / FORMAL_NAME
            atomic_json(formal_dir / "selected_image_candidates.json", plans["formal"])
            downloader.state["small_subset_complete"] = True
            downloader.checkpoint()
            atomic_json(logs / "evqa_gldv2_1898_61_summary.json",
                        {"status": "downloading", "complete": False,
                         "requested_train": 1898, "requested_test": 61,
                         "unique_images": 1812, "subset": FORMAL_NAME,
                         "progress_path": str(downloader.progress_path)})
            formal_download = downloader.run(plans["formal"])
            formal = build_candidate_subset(root, plans["formal"])
            formal["downloads"] = formal_download
            atomic_json(logs / "evqa_gldv2_1898_61_summary.json", formal)
            downloader.state.update(status="complete" if formal["complete"] else "incomplete",
                                    formal_subset_complete=formal["complete"])
            return 0 if formal["complete"] else 1
        except (SafeStop, ValueError, OSError) as error:
            downloader.state.update(status="stopped", stop_reason=str(error))
            downloader.emit("safe_stop", reason=str(error))
            atomic_json(logs / "gldv2_stop_report.json", downloader.state)
            return 2
        finally:
            downloader.checkpoint()
            downloader.session.close()


if __name__ == "__main__":
    raise SystemExit(main())
