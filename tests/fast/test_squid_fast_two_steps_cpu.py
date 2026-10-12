"""Squid fast mode end to end on the CPU (Root REQ_401 steps 1-6 integrated): two steps of cases/squid_fast.py on the
full production graded grid with the D2 band — body, graded kernel, face mask, shifted-penalty air chamber and the
PHYSICAL rigid contact (no ideal check valve: user 2026-10-12) wired as in contract 01 §6; every check runs each step.
Also the Root 2026-10-12 run log (valve label, rim support, penetration, per-body volume error, flap band lift) and the
snapshots.  Integration test only (UNVALIDATED physics)."""
import csv
import json

import numpy as np

import os

import pytest

from cases import squid_fast

# FastFsi export: the squid geometry and its simulation_config.json are not distributed.
pytestmark = pytest.mark.skipif(not os.environ.get('SQUID_SOURCE_CONFIG'),
                                reason='set SQUID_SOURCE_CONFIG to the squid simulation_config.json')

LABELS = {"no_flow", "open", "contact_closure", "partial_contact_closure_film_numerical_suspected",
          "film_closure_numerical_suspected"}


def test_squid_fast_two_steps(tmp_path):
    out = tmp_path / 'run'
    squid_fast.main(['--output', str(out), '--arch', 'cpu', '--max-steps', '2', '--end-time', '1.0',
                     '--row-interval', '1', '--check-interval', '1', '--snapshot-interval-s', '1e-5'])
    final = json.loads((out / 'final.json').read_text())
    assert final['status'] == 'COMPLETED', final
    with (out / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        assert float(row['min_D']) > 0.0
        assert abs(float(row['volume_error'])) < 0.01
        assert 0.0 <= float(row['rim_contact_fraction']) <= 1.0
        assert float(row['contact_force_n']) >= 0.0
        assert float(row['flap_rim_min_centre_gap_m']) > 0.0
        assert row['valve_label'] in LABELS
        assert float(row['min_D_rim_free']) > 0.0
        assert abs(float(row['volume_error_flap'])) < 0.01 and abs(float(row['volume_error_membrane'])) < 0.01
        assert float(row['tether_max_disp_flap_m']) >= 0.0
        assert float(row['max_penetration_m']) >= 0.0
    valve = [json.loads(line) for line in (out / 'valve_log.jsonl').read_text().splitlines()]
    assert valve and valve[0]['label'] == rows[0]['valve_label']
    meta = json.loads((out / 'run_meta.json').read_text())
    assert meta['valve_log']['rim_support_stop_min_D'] == 0.1
    assert meta['contact']['d_c_rim_m'] > 0.0
    assert meta['reachability_flood']['batch_levels'] == 0 and meta['reachability_flood']['bfs_levels'] > 0
    assert 'refill annulus 5.0 mm vs CAD 2.0 mm' in meta['geometry_note']                 # Root REQ_410
    assert json.loads((out / 'final.json').read_text())['geometry_note'] == meta['geometry_note']
    shares = meta['free_volume_shares']
    assert abs(shares['flap_share'] + shares['membrane_share'] - 1.0) < 1e-12 and 0.0 < shares['flap_share'] < 1.0
    static = np.load(out / 'snapshots' / 'static.npz')
    snaps = sorted((out / 'snapshots').glob('snap_*.npz'))
    assert len(snaps) == 2
    snap = np.load(snaps[-1])
    assert snap['u'].shape == static['X0'].shape and snap['J'].shape == static['body_id'].shape
    assert snap['vel_xz'].shape[-1] == 3 and snap['p_xz'].shape == static['obstacle_xz'].shape
    assert np.isfinite(snap['u']).all() and np.isfinite(snap['J']).all()
    print('TWO_STEPS', json.dumps(rows))
