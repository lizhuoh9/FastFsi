"""Stable integer compaction for the existing eighteen ordered report sums.

No value-based dropping: a node is retained whenever any original membership
bit is set, including present signed zeros and nonfinite values. Each slot
still folds the identical subsequence in original (i,j,k) order from +0 f64.
"""
import math
import taichi as ti


@ti.data_oriented
class DivergenceFold:
    BLOCK = 256

    def __init__(self, shape):
        self.shape = tuple(int(v) for v in shape)
        self.capacity = math.prod(self.shape)
        if not 0 < self.capacity <= 2**31 - 1:
            raise ValueError('divergence compaction requires an i32-capacity grid')
        self.blocks = (self.capacity + self.BLOCK - 1) // self.BLOCK
        self.counts = ti.field(ti.i32, shape=self.blocks)
        self.offsets = ti.field(ti.i32, shape=self.blocks)
        self.indices = ti.field(ti.i32, shape=self.capacity)
        self.count = ti.field(ti.i32, shape=())
        self.scratch_bytes = 4 * (self.capacity + 2 * self.blocks + 1)

    @ti.func
    def xyz(self, flat):
        k = flat % self.shape[2]
        j = (flat // self.shape[2]) % self.shape[1]
        i = flat // (self.shape[1] * self.shape[2])
        return ti.Vector([i, j, k])

    @ti.kernel
    def compact(self, owner: ti.template()):
        for block in range(self.blocks):
            begin = block * self.BLOCK
            end = begin + ti.min(self.BLOCK, self.capacity - begin)
            count = 0
            for flat in range(begin, end):
                index = self.xyz(flat)
                if owner._ordered_divergence_membership[index] != ti.u32(0):
                    count += 1
            self.counts[block] = count
        ti.loop_config(serialize=True)
        for unused in range(1):
            offset = 0
            for block in range(self.blocks):
                self.offsets[block] = offset
                offset += self.counts[block]
            self.count[None] = offset
        for block in range(self.blocks):
            begin = block * self.BLOCK
            end = begin + ti.min(self.BLOCK, self.capacity - begin)
            offset = self.offsets[block]
            for flat in range(begin, end):
                index = self.xyz(flat)
                if owner._ordered_divergence_membership[index] != ti.u32(0):
                    self.indices[offset] = flat
                    offset += 1
        # Retired tail cannot retain an earlier trial's indices as evidence.
        for slot in range(self.count[None], self.capacity):
            self.indices[slot] = -1

    @ti.kernel
    def fold(self, owner: ti.template()):
        for slot in range(18):
            total = ti.cast(0.0, ti.f64)
            for position in range(self.count[None]):
                index = self.xyz(self.indices[position])
                if (owner._ordered_divergence_membership[index]
                        & (ti.u32(1) << slot)) != ti.u32(0):
                    total += owner._ordered_divergence_square[index][slot & 1]
            owner.divergence_combined_sum[slot] = total


def fold_report(self):
    compact = self._h2_divergence_fold
    compact.compact(self)
    compact.fold(self)
