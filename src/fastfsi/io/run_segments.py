"""Join the output directories of a run continued by checkpoint / restart into one result directory (create-only).

A run is a chain of SEGMENTS: the first starts at t = 0; every later segment was started with --restart-from a
checkpoint written by the previous one and holds restart.json (written by the driver at the restart):
    {"checkpoint": path, "checkpoint_sha256": ..., "step": S, "t_s": T, "t_text": the history text of T,
     "source_output": the previous segment's directory, "diagnostic": false}
The join, for segment i >= 2 with restart step S_i:
  history  the previous segment's rows up to and including step S_i, then segment i's rows (all of them for the last
           segment).  Checked: identical headers; the previous segment has a row at S_i whose t is the checkpoint's t;
           segment i's first row is the next row (step S_i + row interval, or the run's final step) and later than T_i;
           steps strictly increasing, no duplicate and no missing row (every step difference equals the row interval
           except into the final row); every CUMULATIVE column (all-time totals / maxima) never decreases across the
           boundary;
  vtk      new .pvd collections in OUT_DIR that reference every segment's data files in place (relative paths), frames
           cut the same way by step; a (kind, step) present twice is an error;
  logs     every *.jsonl event log present in a segment, concatenated with the same cut (by "step", else by "t_s");
           lines without either are kept;
  report   OUT_DIR/join_report.json: the segments, cuts, row and frame counts and every check.
A DIAGNOSTIC segment (restart.json "diagnostic": true, a --diagnostic-restart override) is refused unless
--allow-diagnostic, and then OUT_DIR, the history header and the report are marked DIAGNOSTIC.
usage: python -m fastfsi.io.run_segments OUT_DIR SEGMENT_DIR SEGMENT_DIR [...] --row-interval N
                                          [--cumulative-columns a,b,...] [--allow-diagnostic]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from fastfsi.io.vtk import read_index, write_pvds

HISTORY = "history.csv"
RESTART = "restart.json"


class JoinError(ValueError):
    pass


def read_history(directory):
    with open(Path(directory) / HISTORY, newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise JoinError("%s has no header" % (Path(directory) / HISTORY))
    return rows[0], rows[1:]


def _column(header, name):
    if name not in header:
        raise JoinError("history has no column %r" % name)
    return header.index(name)


def join_histories(segments, restarts, row_interval, cumulative):
    """(header, rows, checks) of the joined history; segments: directories; restarts[i]: restart.json of segment i
    (None for the first)."""
    header, rows = None, []
    checks = []
    tables = [read_history(d) for d in segments]
    for index, (seg_header, seg_rows) in enumerate(tables):
        if header is None:
            header = seg_header
        elif seg_header != header:
            raise JoinError("segment %d header differs from segment 0" % index)
    step_col, t_col = _column(header, "step"), _column(header, "t_s")
    cumulative_cols = [(name, _column(header, name)) for name in cumulative]
    for index, (_, seg_rows) in enumerate(tables):
        if index + 1 < len(tables):
            cut = int(restarts[index + 1]["step"])
            kept = [r for r in seg_rows if int(r[step_col]) <= cut]
            if not kept or int(kept[-1][step_col]) != cut:
                raise JoinError("segment %d has no row at the checkpoint step %d" % (index, cut))
            t_text = restarts[index + 1].get("t_text")
            if t_text is not None and kept[-1][t_col] != t_text:
                raise JoinError("segment %d row at step %d has t %s, the checkpoint says %s"
                                % (index, cut, kept[-1][t_col], t_text))
        else:
            kept = list(seg_rows)
        if index > 0:
            cut = int(restarts[index]["step"])
            if not kept:
                raise JoinError("segment %d has no rows" % index)
            first = int(kept[0][step_col])
            if first <= cut:
                raise JoinError("segment %d starts at step %d, not after the checkpoint step %d" % (index, first, cut))
            if float(kept[0][t_col]) <= float(restarts[index]["t_s"]):
                raise JoinError("segment %d first row t %s is not after the checkpoint t %r"
                                % (index, kept[0][t_col], restarts[index]["t_s"]))
            previous = rows[-1]
            for name, col in cumulative_cols:
                before, after = float(previous[col]), float(kept[0][col])
                if after < before:
                    raise JoinError("cumulative column %s decreases across the boundary into segment %d: %r -> %r"
                                    % (name, index, before, after))
                checks.append(dict(boundary=index, column=name, before=before, after=after))
        rows.extend(kept)
    steps = [int(r[step_col]) for r in rows]
    for a, b in zip(steps, steps[1:]):
        if b <= a:
            raise JoinError("steps not increasing: %d then %d" % (a, b))
    gaps = [(a, b) for a, b in zip(steps, steps[1:]) if b - a != row_interval]
    if gaps and (len(gaps) > 1 or gaps[0][1] != steps[-1]):
        raise JoinError("missing or extra rows (row interval %d): %s" % (row_interval, gaps[:5]))
    times = [float(r[t_col]) for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise JoinError("t not increasing")
    return header, rows, checks


def join_vtk(segments, restarts):
    entries, counts = [], []
    seen = set()
    for index, directory in enumerate(segments):
        index_path = Path(directory) / "vtk" / "vtk_series.jsonl"
        if not index_path.is_file():
            counts.append(0)
            continue
        seg_entries, skipped = read_index(index_path)
        if skipped:
            raise JoinError("segment %d vtk index has %d unreadable lines" % (index, skipped))
        if index + 1 < len(segments):
            cut = int(restarts[index + 1]["step"])
            seg_entries = [e for e in seg_entries if int(e["step"]) <= cut]
        for e in seg_entries:
            key = (e["kind"], int(e["step"]))
            if key in seen:
                raise JoinError("vtk frame %s step %d appears twice" % key)
            seen.add(key)
            entries.append(dict(e, file=str((index_path.parent / e["file"]).resolve())))
        counts.append(len(seg_entries))
    return entries, counts


def _cut_key(line):
    if "step" in line and line["step"] is not None:
        return "step", int(line["step"])
    if "t_s" in line and line["t_s"] is not None:
        return "t_s", float(line["t_s"])
    return None, None


def join_logs(segments, restarts, out_dir):
    names = sorted({p.name for d in segments for p in Path(d).glob("*.jsonl")})
    report = {}
    for name in names:
        lines_out, per_segment = [], []
        for index, directory in enumerate(segments):
            path = Path(directory) / name
            if not path.is_file():
                per_segment.append(0)
                continue
            kept = 0
            for text in path.read_text(encoding="utf-8").splitlines():
                if not text.strip():
                    continue
                if index + 1 < len(segments):
                    kind, value = _cut_key(json.loads(text))
                    limit = (int(restarts[index + 1]["step"]) if kind == "step"
                             else float(restarts[index + 1]["t_s"]) if kind == "t_s" else None)
                    if limit is not None and value > limit:
                        continue
                lines_out.append(text)
                kept += 1
            per_segment.append(kept)
        with open(out_dir / name, "x", encoding="utf-8") as stream:
            stream.write("".join(line + "\n" for line in lines_out))
        report[name] = per_segment
    return report


def join(out_dir, segments, row_interval, cumulative=(), allow_diagnostic=False):
    out_dir = Path(out_dir)
    segments = [Path(s) for s in segments]
    if len(segments) < 2:
        raise JoinError("a join needs at least two segments")
    restarts = [None]
    for index, directory in enumerate(segments[1:], start=1):
        path = directory / RESTART
        if not path.is_file():
            raise JoinError("segment %d (%s) has no %s" % (index, directory, RESTART))
        restart = json.loads(path.read_text(encoding="utf-8"))
        if Path(restart["source_output"]).resolve() != segments[index - 1].resolve():
            raise JoinError("segment %d restarted from %s, not from segment %d (%s)"
                            % (index, restart["source_output"], index - 1, segments[index - 1]))
        restarts.append(restart)
    if (segments[0] / RESTART).is_file():
        raise JoinError("the first segment %s is itself a restart; join from the t = 0 segment" % segments[0])
    diagnostic = any(r.get("diagnostic") for r in restarts[1:])
    if diagnostic and not allow_diagnostic:
        raise JoinError("a segment is a DIAGNOSTIC restart (code or library override); its outputs are never results")
    header, rows, checks = join_histories(segments, restarts, int(row_interval), list(cumulative))
    entries, frame_counts = join_vtk(segments, restarts)
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / HISTORY, "x", newline="") as stream:
        writer = csv.writer(stream)
        if diagnostic:
            stream.write("# DIAGNOSTIC: joined from a --diagnostic-restart segment; never a result\n")
        writer.writerow(header)
        writer.writerows(rows)
    pvds = write_pvds(entries, Path("/"), out_dir, comment="joined from %d segments%s"
                      % (len(segments), " (DIAGNOSTIC)" if diagnostic else "")) if entries else []
    logs = join_logs(segments, restarts, out_dir)
    step_col = header.index("step")
    report = dict(diagnostic=diagnostic, segments=[str(s.resolve()) for s in segments],
                  restarts=restarts[1:], rows=len(rows), first_step=int(rows[0][step_col]) if rows else None,
                  last_step=int(rows[-1][step_col]) if rows else None, row_interval=int(row_interval),
                  cumulative_checks=checks, vtk_frames_per_segment=frame_counts, pvds=[str(p) for p in pvds],
                  logs=logs)
    with open(out_dir / "join_report.json", "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir")
    p.add_argument("segments", nargs="+")
    p.add_argument("--row-interval", type=int, required=True)
    p.add_argument("--cumulative-columns", default="")
    p.add_argument("--allow-diagnostic", action="store_true")
    a = p.parse_args(argv)
    cumulative = [c for c in a.cumulative_columns.split(",") if c]
    try:
        report = join(a.out_dir, a.segments, a.row_interval, cumulative, a.allow_diagnostic)
    except JoinError as error:
        print("JOIN REFUSED: %s" % error, file=sys.stderr)
        return 2
    print("JOINED %d segments: %d rows (steps %s..%s), vtk frames %s, logs %s%s"
          % (len(report["segments"]), report["rows"], report["first_step"], report["last_step"],
             report["vtk_frames_per_segment"], sorted(report["logs"]), " DIAGNOSTIC" if report["diagnostic"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
