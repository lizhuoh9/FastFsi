"""Checkpoint round trip of real Taichi fields on the CPU: 0-d, scalar, vector and matrix fields (bitwise)."""
import numpy as np
import pytest
import taichi as ti

from fastfsi.io.checkpoint import Checkpointable, load_components, read_checkpoint, write_checkpoint


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, offline_cache=False)


@ti.data_oriented
class Component(Checkpointable):
    CHECKPOINT_FIELDS = ("count", "values", "vectors", "matrices", "wide")
    CHECKPOINT_HOST = ("steps", "seen")

    def __init__(self):
        self.count = ti.field(ti.i32, shape=())
        self.values = ti.field(ti.f64, shape=(3, 4))
        self.vectors = ti.Vector.field(3, ti.f32, shape=5)
        self.matrices = ti.Matrix.field(3, 3, ti.f32, shape=2)
        self.wide = ti.field(ti.u8, shape=(2, 2, 2))
        self.steps, self.seen = 0, set()


def fill(c, seed):
    rng = np.random.default_rng(seed)
    c.count[None] = int(rng.integers(1, 100))
    c.values.from_numpy(rng.standard_normal((3, 4)))
    vectors = rng.standard_normal((5, 3)).astype(np.float32)
    vectors[0, 0] = np.nan
    c.vectors.from_numpy(vectors)
    c.matrices.from_numpy(rng.standard_normal((2, 3, 3)).astype(np.float32))
    c.wide.from_numpy(rng.integers(0, 255, (2, 2, 2)).astype(np.uint8))
    c.steps, c.seen = 17, {(1, 2), (3, 4)}


def test_taichi_fields_round_trip_bitwise(tmp_path):
    saved = Component()
    fill(saved, 1)
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(c=saved), dict(step=3), {})
    arrays, host, loop, _, header = read_checkpoint(path)
    assert {e["name"]: e["shape"] for e in header["arrays"]}["count"] == []
    fresh = Component()
    load_components(dict(c=fresh), arrays, host)
    for name in Component.CHECKPOINT_FIELDS:
        a, b = getattr(saved, name).to_numpy(), getattr(fresh, name).to_numpy()
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), name
    assert fresh.steps == 17 and fresh.seen == {(1, 2), (3, 4)} and loop == dict(step=3)
