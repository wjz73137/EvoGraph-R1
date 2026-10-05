import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_targeted_training_source_activates_graph_edit_routing(tmp_path):
    rows = [
        {'data_source': 'E-VQA/graph_edit_validation', 'data_id': 'val:0',
         'extra_info': {'question': 'How many stars does this hotel have?', 'image_id': 'hotel', 'split': 'test'}},
        {'data_source': 'E-VQA/graph_edit_validation', 'data_id': 'val:1',
         'extra_info': {'question': 'In which country is this lake located?', 'image_id': 'lake', 'split': 'test'}},
    ]
    source = tmp_path / 'test.parquet'
    output = tmp_path / 'targeted'
    pq.write_table(pa.Table.from_pylist(rows), source)
    script = Path(__file__).resolve().parents[2] / 'scripts/prepare_evqa_graphedit_targeted_smoke.py'
    subprocess.run([sys.executable, str(script), '--source', str(source), '--output', str(output)], check=True)
    generated = pq.read_table(output / 'train.parquet').to_pylist()
    assert len(generated) == 2
    assert all('graph_edit' in row['data_source'] for row in generated)
    assert all(row['extra_info']['split'] == 'train' for row in generated)
    assert pq.read_table(output / 'test.parquet').to_pylist() == rows
