"""Finite caller for an independently source-verified native CG window.

The provider owns record/replay and original bookkeeping. Missing/unsupported
providers fall back before any writes; started failures are never swallowed.
"""
from simulation_core.fluids.shared_definitions import _CURRENT


def nonpoll_window(owner, *, first_iteration, max_iters, outlet, authority,
                   projector, use_mg, project_components, remove_mean,
                   apply_preconditioner, project_component_means):
    remaining = min(15, 16 - first_iteration % 16, max_iters - first_iteration)
    if first_iteration % 16 == 0 or remaining <= 0:
        return 0
    # A live cold setup budget has per-stage deadline observations. Graph
    # replay must not bypass them. The ordinary native sequence remains live.
    if _CURRENT.get() is not None:
        return 0
    session = getattr(owner, '_k7_active_session', None)
    provider = getattr(session, 'cg_nonpoll_window', None)
    if provider is None:
        return 0
    completed = provider(owner, first_iteration=first_iteration, count=remaining,
        max_iters=max_iters, outlet=outlet, authority=authority, projector=projector,
        use_mg=bool(use_mg), project_components=bool(project_components),
        remove_mean=bool(remove_mean), apply_preconditioner=apply_preconditioner,
        project_component_means=project_component_means)
    if completed is not True and completed is not False:
        raise TypeError('CG window provider must return the finite bool ABI')
    return remaining if completed else 0
