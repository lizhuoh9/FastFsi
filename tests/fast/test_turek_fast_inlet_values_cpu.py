"""F3 item #1 (Root REQ_387/388; proto_04): the values-only inlet update of cases/turek_fast.py (after the first full inlet
write, only the inlet face values change; no graph / ledger / reachability invalidation and no re-marking) gives the
same history rows as the old per-step full rewrite with reachability re-marking, every column except the wall time,
during the inlet ramp (four steps, the first one full in both runs)."""
import csv

from cases import turek_fast


def run(tmp_path, mode):
    out = tmp_path / mode
    turek_fast.main(['--preset', 'fsi3', '--grid', '4,24,144', '--end-time', '0.01', '--max-steps', '4',
                     '--output', str(out), '--arch', 'cpu', '--time-integration', 'midpoint', '--inlet-update', mode])
    with (out / 'history.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_values_only_inlet_update_matches_the_full_rewrite(tmp_path):
    full = run(tmp_path, 'full')
    values = run(tmp_path, 'values')
    assert len(full) == len(values) == 4
    for row_full, row_values in zip(full, values):
        for key, value in row_full.items():
            if key != 'wall_s_per_step':
                assert row_values[key] == value, (row_full['step'], key, value, row_values[key])
