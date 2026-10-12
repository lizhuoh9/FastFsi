"""Squid fast driver wiring of Helper 1's hypre hook (Root REQ_407): two CPU steps with --pressure-solver hypre_pfmg
and the SDF cache directory.  The hook must take every eligible pressure solve (no fallback), the solver's own CG must
confirm with 0 iterations, the setup must record the phases (peak RSS per phase) and the SDF cache file must be written.
Integration test only (UNVALIDATED physics; the CG / hypre equivalence is checked on the card, Root's criterion)."""
import csv
import json

import os

import pytest

from cases import squid_fast

# FastFsi export: the squid geometry and its simulation_config.json are not distributed.
pytestmark = pytest.mark.skipif(not os.environ.get('SQUID_SOURCE_CONFIG'),
                                reason='set SQUID_SOURCE_CONFIG to the squid simulation_config.json')


def test_squid_fast_two_steps_hypre(tmp_path):
    out = tmp_path / 'run'
    cache = tmp_path / 'sdf_cache'
    squid_fast.main(['--output', str(out), '--arch', 'cpu', '--max-steps', '2', '--end-time', '1.0',
                     '--row-interval', '1', '--check-interval', '1', '--pressure-solver', 'hypre_pfmg',
                     '--sdf-cache-dir', str(cache)])
    final = json.loads((out / 'final.json').read_text())
    assert final['status'] == 'COMPLETED', final
    totals = final['hypre']['totals']
    assert totals['hypre_calls'] > 0 and sum(totals['fallbacks'].values()) == 0, totals
    with (out / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        assert row['pressure_solver'] == 'hypre_pfmg'
        assert int(row['cg_iterations']) == 0                 # the CG only confirms the hypre solution
        assert float(row['pressure_iterations_per_step']) > 0.0
        assert int(row['hypre_fallbacks']) == 0
        assert float(row['div_water_max_1ps']) >= 0.0
    setup = json.loads((out / 'setup_report.json').read_text())
    phases = [r['phase'] for r in setup['setup_phases']]
    assert phases[:4] == ['inputs', 'obstacle', 'particles', 'taichi_init'] and 'contact' in phases
    assert 'hypre_hook' in phases and phases[-1] == 'initial_min_D'
    assert all(r['peak_rss_bytes'] > 0 for r in setup['setup_phases'])
    assert len(list(cache.glob('sdf_*.npz'))) == 1
    meta = json.loads((out / 'run_meta.json').read_text())
    assert meta['pressure_solver']['name'] == 'hypre_pfmg'
    print('TWO_STEPS_HYPRE', json.dumps(rows))
