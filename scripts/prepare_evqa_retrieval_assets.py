#!/usr/bin/env python3
"""Resume the official E-VQA wiki KB and download the configured GME snapshot.

No images, packages, credentials, GPU workloads, or system settings are changed.
The original QA files are read-only. KB selection uses URLs, never answers.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json, publish_image
from scripts import run_evqa_gpu_smoke as smoke
from scripts.download_gldv2_thumbnails import now, retry_after_seconds

ROOT = smoke.ROOT
LOGS = ROOT / 'logs'
KB = ROOT / 'datasets_mm/E-VQA/raw/kb'
KB_URL = 'https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip'
JSON_SHA256 = '36af1b6718a975c355a776114be216f4800c61320897b2186d33d17a08e44c77'
GME_PATH = Path('/home/data/dataset/wjz/models/gme-Qwen2-VL-2B-Instruct')
GME_REPO = 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'
GME_REVISION = '9cfa6413f704a7c1cf5064d240748e10c876b286'


def digest_file(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def requested_urls():
    subset = ROOT / 'datasets_mm/E-VQA/subsets' / smoke.SUBSET
    result = set()
    for split, expected in [('train', 64), ('test', 16)]:
        with (subset/f'qa_{split}.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != expected:
            raise ValueError('original small split counts changed')
        for row in rows:
            result.update(u.strip() for u in row['wikipedia_url'].split('|') if u.strip())
    if not result:
        raise ValueError('no reliable Wikipedia URL mapping')
    return result


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def extract_pages(archive_path, checkpoint):
    import ijson
    wanted = requested_urls()
    pages = {}
    output = KB / 'paper_64_16_seed0_wiki_pages.json'
    if output.exists():
        previous = json.loads(output.read_text())
        if (previous.get('source_json_sha256') == JSON_SHA256
                and previous.get('requested_urls') == sorted(wanted)
                and previous.get('complete')):
            return previous
        raise ValueError('preserving existing different wiki page extraction')
    with zipfile.ZipFile(archive_path) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()
                   and i.filename.endswith('.json')]
        if len(members) != 1:
            raise ValueError('official KB archive does not have exactly one JSON member')
        member = members[0]
        checkpoint(phase='streaming_verified_wiki_json', json_bytes_total=member.file_size)
        with archive.open(member) as content:
            reader = HashingReader(content)
            processed = 0
            # Read to EOF even after finding all requested URLs: verify ZIP CRC
            # and the official full JSON SHA-256, without unpacking the full JSON.
            for url, page in ijson.kvitems(reader, ''):
                processed += 1
                if url in wanted:
                    if not isinstance(page, dict) or not isinstance(page.get('section_texts'), list):
                        raise ValueError('wiki page has unexpected schema')
                    if page.get('url') and page['url'] != url:
                        raise ValueError('wiki URL identity mismatch')
                    pages[url] = page
                if processed % 10000 == 0:
                    checkpoint(pages_scanned=processed, pages_found=len(pages),
                               json_bytes_read=reader.bytes)
            if reader.digest.hexdigest() != JSON_SHA256:
                raise ValueError('official uncompressed JSON SHA-256 mismatch')
    result = {'complete': wanted == pages.keys(), 'source_url': KB_URL,
              'source_zip': str(archive_path), 'source_json_sha256': JSON_SHA256,
              'zip_member': member.filename, 'requested_urls': sorted(wanted),
              'missing_urls': sorted(wanted-pages.keys()), 'pages': pages,
              'selected_by': 'existing fixed QA Wikipedia URLs; no answer-based selection',
              'extracted_at': now()}
    atomic_json(output, result)
    return result


def prepare_kb(checkpoint):
    KB.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.trust_env = False
        session.headers['User-Agent'] = 'EvoGraph-EVQA-Academic-Downloader/1.0'
        with session.head(KB_URL, timeout=(15, 60), allow_redirects=False) as response:
            response.raise_for_status()
            total = int(response.headers['Content-Length'])
            etag = response.headers['ETag']
            hashes = dict(h.split('=', 1) for h in response.headers['x-goog-hash'].split(', '))
            md5 = base64.b64decode(hashes['md5']).hex()
        target = KB/'encyclopedic_kb_wiki.zip'
        part = target.with_suffix('.zip.part')
        checkpoint(phase='checking_kb_resume', total_bytes=total, remote_etag=etag,
                   expected_zip_md5=md5)
        if target.exists():
            if target.stat().st_size != total or digest_file(target, 'md5') != md5:
                raise ValueError('preserving existing different KB archive')
        else:
            if part.exists() and part.stat().st_size:
                size = part.stat().st_size
                if size > total:
                    raise ValueError('existing partial KB is larger than source; preserved')
                for start, end in [(0, min(4095,size-1)), (max(0,size-4096),size-1)]:
                    with session.get(KB_URL, headers={'Range':f'bytes={start}-{end}',
                                                     'If-Match':etag}, timeout=(15,60)) as r:
                        if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                            raise ValueError('KB source does not honor identity-checked byte ranges')
                        with part.open('rb') as old:
                            old.seek(start)
                            if old.read(end-start+1) != r.content:
                                raise ValueError('existing KB partial differs from official source; preserved')
            for attempt in range(4):
                size = part.stat().st_size if part.exists() else 0
                if size == total:
                    break
                if shutil.disk_usage(ROOT).free < max(100_000_000_000, total-size):
                    raise ValueError('free disk space below safety threshold')
                checkpoint(phase='downloading_kb', bytes_downloaded=size, total_bytes=total)
                try:
                    with session.get(KB_URL, headers={'Range':f'bytes={size}-', 'If-Match':etag},
                                     timeout=(15,60), stream=True, allow_redirects=False) as r:
                        if r.status_code == 429:
                            delay=retry_after_seconds(r.headers.get('Retry-After'))
                            time.sleep(delay if delay is not None else [60,300,900][min(attempt,2)])
                            continue
                        if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {size}-{total-1}/{total}':
                            raise ValueError('download response has unexpected status or byte range')
                        last_checkpoint=time.monotonic()
                        with part.open('ab') as out:
                            for chunk in r.iter_content(4*1024*1024):
                                if not chunk:
                                    continue
                                if size+len(chunk)>total:
                                    raise ValueError('download exceeds declared archive size')
                                out.write(chunk); size+=len(chunk)
                                if time.monotonic()-last_checkpoint>=15:
                                    out.flush()
                                    checkpoint(bytes_downloaded=size)
                                    last_checkpoint=time.monotonic()
                            out.flush(); os.fsync(out.fileno())
                    break
                except requests.RequestException as error:
                    checkpoint(network_error=type(error).__name__, bytes_downloaded=part.stat().st_size)
                    if attempt==3:
                        raise
                    time.sleep([30,60,120][attempt])
            if part.stat().st_size != total or digest_file(part,'md5') != md5:
                raise ValueError('KB zip size/MD5 verification failed; partial file preserved')
            if not zipfile.is_zipfile(part):
                raise ValueError('KB archive is not a valid ZIP; preserved')
            publish_image(part, target)
        checkpoint(phase='kb_archive_verified', zip_md5_valid=True, bytes_downloaded=total)
    extracted=extract_pages(target,checkpoint)
    return {'archive':str(target),'bytes':total,'zip_md5_valid':True,
            'json_sha256_valid':True,'selected_pages':len(extracted['pages']),
            'missing_urls':extracted['missing_urls'],'complete':extracted['complete'],
            'pages_file':str(KB/'paper_64_16_seed0_wiki_pages.json')}


def prepare_embedding(checkpoint):
    smoke.MODEL = GME_PATH
    checkpoint(phase='downloading_gme_embedding', repo=GME_REPO, revision=GME_REVISION)
    smoke.run_hf(['download',GME_REPO,'--revision',GME_REVISION,'--local-dir',str(GME_PATH),
                  '--max-workers','1','--quiet'])
    checkpoint(phase='verifying_gme_checksums')
    smoke.run_hf(['cache','verify',GME_REPO,'--revision',GME_REVISION,
                  '--local-dir',str(GME_PATH),'--fail-on-missing-files'])
    return {'complete':True,'repo':GME_REPO,'revision':GME_REVISION,
            'path':str(GME_PATH),'checksums_verified':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['kb','embedding'])
    args=parser.parse_args(argv)
    LOGS.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'stage':args.stage,'started_at':now(),'status':'running',
           'gpu_used':False,'training_started':False,'api_keys_used':False}
    path=LOGS/f'evqa_{args.stage}_assets_progress.json'
    def checkpoint(**updates):
        state.update(**updates,updated_at=now())
        atomic_json(path,state)
    with (LOGS/f'.evqa_{args.stage}_assets.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            checkpoint(phase='preflight')
            if shutil.disk_usage(ROOT).free<100_000_000_000:
                raise ValueError('free disk space below 100 GB')
            result=prepare_kb(checkpoint) if args.stage=='kb' else prepare_embedding(checkpoint)
            checkpoint(status='complete' if result['complete'] else 'incomplete',
                       result=result,finished_at=now())
        except Exception as error:
            checkpoint(status='failed',error_type=type(error).__name__,
                       error=smoke.sanitized(error),finished_at=now())
        atomic_json(LOGS/f'evqa_{args.stage}_assets_report.json',state)
        print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
    return 0 if state['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
