"""Squid fast-mode setup (cases/squid_fast.py --setup-only; Root REQ_401 step 6) on the CPU: the production graded grid
with the D2 band, the static obstacle + groove, the material points (0.5 Delta per axis), tethers and tags, the
production fluid with the sealed ledger and z-min reachability; the points inside rigid-wall cells are dropped."""
import json

import os

import pytest

from cases import squid_fast

# FastFsi export: the squid geometry and its simulation_config.json are not distributed.
pytestmark = pytest.mark.skipif(not os.environ.get('SQUID_SOURCE_CONFIG'),
                                reason='set SQUID_SOURCE_CONFIG to the squid simulation_config.json')


def test_squid_fast_setup_only(tmp_path):
    out = tmp_path / 'setup'
    squid_fast.main(['--output', str(out), '--arch', 'cpu', '--setup-only'])
    final = json.loads((out / 'final.json').read_text())
    assert final['status'] == 'SETUP_ONLY'
    report = json.loads((out / 'setup_report.json').read_text())
    assert report['grid']['nodes'][0] == 92 and report['grid']['nodes'][1] == 92
    assert report['grid']['cells'] > 1_500_000
    assert report['particles']['tethered'] > 0
    assert report['particles']['air_face_points'] > 0 and report['particles']['valve_rim_points'] > 0
    assert report['dropped']['dropped_by_body']['1'] == 0
    assert report['dropped']['dropped_points'] == report['dropped']['dropped_tethered']
    assert report['fluid']['unreached_cells'] == 0
    print('SETUP_REPORT', json.dumps({k: report[k] for k in ('grid', 'dropped', 'fluid', 'dt_elastic_s', 'h_min_m',
                                                             'obstacle_cells', 'setup_wall_s')}))
