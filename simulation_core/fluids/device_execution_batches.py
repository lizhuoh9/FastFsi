"""Isolated selection of the existing, unchanged P2 mean compositions.

No runtime import, field allocation, new reduction, or solver control lives here.
The native branch caller still validates its graph and handles zero components.
"""
from dataclasses import dataclass
from typing import Any, Callable

MEAN_SEQUENCES = {
    "unreached": ("_accumulate_unreached_component_means_kernel",
                  "_subtract_unreached_component_means_kernel",
                  "_p2_unreached_means_kernel"),
    "operator": ("_accumulate_pressure_outlet_operator_component_means_kernel",
                 "_subtract_pressure_outlet_operator_component_means_kernel",
                 "_p2_operator_means_kernel"),
    "nullspace": ("_accumulate_pressure_outlet_nullspace_component_means_kernel",
                  "_subtract_pressure_outlet_nullspace_component_means_kernel",
                  "_p2_nullspace_means_kernel"),
}


@dataclass(frozen=True, eq=False)
class MeanSequenceView:
    owner: Any
    generation: Any
    branch: str
    fields: Any
    accumulate: Callable
    subtract: Callable
    composed: Callable
    zero: Callable
    layout_version: int = 1


@dataclass(frozen=True)
class NativeExecutionContext:
    """Opaque references, not new checks/charges or acceptance authority."""
    status: Any
    budget: Any


class DeviceExecutionBatches:
    def __init__(self, *, cg_mean_batch=False):
        self.cg_mean_batch = bool(cg_mean_batch)

    def subtract_means(self, *, view, field, component_count, owner,
                       generation, context):
        if view.layout_version != 1 or view.branch not in MEAN_SEQUENCES:
            raise ValueError("unsupported mean sequence view")
        if owner is not view.owner or generation != view.generation:
            raise RuntimeError("mean sequence native owner/generation changed")
        # Native callers normalize counts and check graph validity before here.
        # Do not speculate a zero-count reduction or bypass the global mean path.
        if component_count <= 0:
            return view.zero(field)
        if self.cg_mean_batch and context.budget is None:
            return view.composed(field, component_count)
        view.accumulate(field, component_count)
        return view.subtract(field, component_count)
