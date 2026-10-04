#!/usr/bin/env python3
"""Serve an identified API-built E-VQA graph, read-only on localhost."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evograph_mm.kb.gldv2_subset import atomic_json
from scripts.run_evqa_api_graph import OUTPUT
from scripts.run_evqa_gpu_smoke import ROOT, SUBSET
from scripts.run_evqa_retrieval_smoke import GME


def visual_scene_payload(output, image_id):
    scene_path = output / 'mm_store/visual/scene_records.json'
    audit_path = output.parent / 'visual_graph_audit.json'
    if not scene_path.is_file() or not audit_path.is_file():
        return None
    scenes = json.loads(scene_path.read_text())
    if image_id not in scenes:
        return None
    audit = json.loads(audit_path.read_text())
    source = next(item for item in audit['images'] if item['image_id'] == image_id)
    owner = json.loads((output / 'owner.json').read_text())
    associated = json.loads((output / 'mm_store/graph/image_anchor_lookup.json').read_text())[image_id]
    return {'image_id': image_id, 'scene': scenes[image_id], 'evidence_basis': 'image_only',
            'image_anchor_node': source['anchor_node'], 'scene_hyperedge_node': source['scene_node'],
            'visual_relation_nodes': source['visual_relation_nodes'],
            'api_model': owner['visual_model'], 'human_reviewed': False, 'all_facts_verified': False,
            'associated_article': associated['canonical_entity'],
            'article_association_is_visual_identity': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8003)
    parser.add_argument('--working-dir', type=Path, default=OUTPUT,
                        help='identified API baseline, cleaned, visual, or strict Max graph')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    output = args.working_dir.resolve()
    allowed = {OUTPUT.resolve(), (ROOT / 'expr_mm/evqa_api_graph_cleaned_v1/E-VQA').resolve(),
               (ROOT / 'expr_mm/evqa_api_visual_graph_v1/E-VQA').resolve(),
               (ROOT / 'expr_mm/evqa_api_strict_max_graph_v1/E-VQA').resolve(),
               (ROOT / 'expr_mm/evqa_api_strict_max_graph_63_v1/E-VQA').resolve(),
               (ROOT / 'expr_mm/evqa_api_strict_max_graph_full1891_v1/E-VQA').resolve()}
    if output not in allowed:
        parser.error('working-dir must be an identified API baseline, cleaned, visual, or strict Max graph')
    report = json.loads((output.parent / 'report.json').read_text())
    indexes_ready = (report.get('indexes_complete') is True or (
        report.get('indexes_built') is True and report.get('index_status') == 'complete'))
    if report['status'] != 'complete' or not indexes_ready:
        raise RuntimeError('API-built graph/indexes are incomplete')
    owner = json.loads((output / 'owner.json').read_text())
    if not owner['graph_llm'].startswith('api/'):
        raise RuntimeError('refusing to serve a locally extracted graph')
    os.environ.update(
        CUDA_VISIBLE_DEVICES='',
        MM_EMBED_DEVICE='cpu',
        EVOGRAPH_MM_EMBED_RUNTIME_DEVICE='cpu',
        EVOGRAPH_MM_ENABLE_BGE_TEXT='1',
        BGE_MODEL_PATH=str(GME.parent / 'bge-large-en-v1.5'),
        BGE_DEVICE='cpu',
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
    )
    service_id = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    lock_path = ROOT / f'logs/.evqa_retrieval_{service_id}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as check:
            check.bind(('127.0.0.1', args.port))
        import torch
        import faiss
        import uvicorn
        from evograph_mm.kb.api import create_app
        from evograph_mm.kb.indexing.encoders import GMEQwen2VLEncoder
        from starlette.responses import JSONResponse
        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        os.chdir(output)
        app = create_app(working_dir=output, model_path=GME, dataset='E-VQA',
                         subset=report.get('subset', SUBSET),
                         encoder_factory=lambda *a, **kw: GMEQwen2VLEncoder(GME, batch_size=8),
                         reload_interval=0)
        status = app.state.mm_api.status()
        if status['status'] != 'ready':
            raise RuntimeError('API-built graph retrieval is not ready')
        @app.get('/visual-scenes/{image_id}')
        async def get_visual_scene(image_id: str):
            payload = visual_scene_payload(output, image_id)
            if payload is None:
                return JSONResponse({'error': 'no API-extracted scene for this image'}, status_code=404)
            return payload
        @app.middleware('http')
        async def retrieval_only(request, call_next):
            if request.method not in {'GET', 'HEAD', 'OPTIONS'} and not (
                    request.method == 'POST' and request.url.path == '/search'):
                return JSONResponse({'error': 'read-only API-built graph service'}, status_code=405)
            return await call_next(request)
        service = {'pid': os.getpid(), 'host': '127.0.0.1', 'port': args.port,
                   'device': 'cpu', 'cpu_threads': 4, 'read_only': True,
                   'working_dir': str(output), 'graph_llm': owner['graph_llm'], 'status': status}
        atomic_json(ROOT / f'logs/evqa_api_graph_retrieval_status_{service_id}.json', service)
        uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
