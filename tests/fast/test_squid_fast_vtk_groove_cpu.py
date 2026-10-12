"""Squid fast driver v7 on the CPU: (1) two steps with the VTK export at every check (Helper 2's vtk_export_fast,
REQ_409/410): fluid 3-D, both axis planes and the material points written, the four .pvd collections after the run,
the index and the series meta carry the run label (UNVALIDATED, solver) and the geometry note; (2) --groove off (Root
T9 A2): the static obstacle is the original CAD wall, exactly groove_carved_cells more rigid cells than with the groove,
and the geometry note says so.  Integration test only (UNVALIDATED physics)."""
import json

import os

import pytest

from cases import squid_fast

# FastFsi export: the squid geometry and its simulation_config.json are not distributed.
pytestmark = pytest.mark.skipif(not os.environ.get('SQUID_SOURCE_CONFIG'),
                                reason='set SQUID_SOURCE_CONFIG to the squid simulation_config.json')


def test_vtk_series_written_by_the_driver(tmp_path):
    out = tmp_path / 'run'
    squid_fast.main(['--output', str(out), '--arch', 'cpu', '--max-steps', '2', '--end-time', '1.0',
                     '--row-interval', '1', '--check-interval', '1', '--vtk-points-interval-s', '1e-5',
                     '--vtk-3d-interval-s', '1e-5'])
    assert json.loads((out / 'final.json').read_text())['status'] == 'COMPLETED'
    vtk = out / 'vtk'
    index = [json.loads(line) for line in (vtk / 'vtk_series.jsonl').read_text().splitlines()]
    kinds = sorted(e['kind'] for e in index)
    assert kinds == sorted(['fluid_3d'] * 2 + ['plane_xz'] * 2 + ['plane_yz'] * 2 + ['points'] * 2), kinds
    for entry in index:
        assert (vtk / entry['file']).is_file() and entry['bytes'] > 0
        assert (vtk / entry['file']).stat().st_size == entry['bytes']
    assert len(list(vtk.glob('*.pvd'))) == 4
    meta = json.loads((vtk / 'vtk_series_meta.json').read_text())
    assert meta['label'].startswith('UNVALIDATED engineering run') and 'solver cg' in meta['label']
    assert 'refill annulus 5.0 mm vs CAD 2.0 mm' in meta['geometry_note']
    run_meta = json.loads((out / 'run_meta.json').read_text())
    assert run_meta['vtk']['points_interval_s'] == 1e-5 and run_meta['run_label'] == meta['label']


def test_groove_off_is_the_original_cad_wall(tmp_path):
    reports = {}
    for groove in ('on', 'off'):
        out = tmp_path / ('setup_' + groove)
        squid_fast.main(['--output', str(out), '--arch', 'cpu', '--setup-only', '--groove', groove])
        reports[groove] = json.loads((out / 'setup_report.json').read_text())
    on, off = reports['on'], reports['off']
    assert off['groove'] is None and off['obstacle'].get('groove') == 'off'
    assert off['obstacle_cells'] - on['obstacle_cells'] == on['groove']['carved_cells'] > 0
    assert off['geometry_note'].startswith('no groove: the original CAD chamber wall (refill annulus 2.0 mm')
