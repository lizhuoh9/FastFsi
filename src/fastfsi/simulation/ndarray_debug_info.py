"""Host overhead of Taichi ndarray allocation: empty call-stack strings in non-debug runtimes.

Taichi 1.7.4 attaches a DebugInfo string to every ndarray allocation (taichi/lang/_ndarray.py ScalarNdarray.__init__
and the vector/matrix ndarrays of taichi/lang/matrix.py): get_traceback() = traceback.extract_stack() +
traceback.format_list(), which walks and re-reads (linecache, one os.stat per frame) every frame of the deep FSI call
stack, and the string is passed to the allocation as its DebugInfo. An FSI3 profile (py-spy)
spent 6.7 s of a 72.1 s step in it. What the tests show: with the empty string the tested allocations keep
their shapes, dtypes and zero fill, and the kernel output bytes are unchanged. What is NOT shown: that no native
error or diagnostic path reads DebugInfo; such a message raised in an initialized non-debug runtime would carry no
Python call stack.

The two allocation modules' get_traceback is replaced by one delegating function that decides at
every call from the actual runtime state. If no program is initialized, or the initialized program is in debug mode,
it returns the original get_traceback with one extra stack level: the identical string, with the delegation frame
excluded. If the initialized program is not in debug mode, it returns ''. A later reset or ti.init(debug=True) is
therefore served by the original capture with no re-installation. init_taichi installs it after a successful
non-debug initialization. Installing is idempotent and all-or-nothing: if either module holds a function that is
neither Taichi's nor this delegate, nothing is replaced and RuntimeError is raised. restore_ndarray_debug_info() puts
the original functions back. Field/SNode declarations
(taichi/lang/impl.py, snode.py) keep the original traceback.
"""
import taichi.lang._ndarray as _ndarray
import taichi.lang.impl as _impl
import taichi.lang.matrix as _matrix
from taichi.lang.util import get_traceback as _taichi_get_traceback

_MODULES = (_ndarray, _matrix)
_ORIGINAL = {}


def _capture_wanted():
    """True when no program is initialized or the initialized program is in debug mode."""
    prog = _impl.get_runtime().prog
    return prog is None or bool(prog.config().debug)


def _delegating_traceback(stacklevel=1):
    if _capture_wanted():
        return _taichi_get_traceback(stacklevel + 1)
    return ''


def use_empty_ndarray_debug_info():
    """Install the delegate in both ndarray allocation modules (all or nothing); returns the module names."""
    for module in _MODULES:
        current = module.get_traceback
        if current is not _delegating_traceback and current is not _taichi_get_traceback:
            raise RuntimeError('unexpected get_traceback in %s; nothing replaced' % module.__name__)
    for module in _MODULES:
        if module.get_traceback is not _delegating_traceback:
            _ORIGINAL.setdefault(module.__name__, module.get_traceback)
            module.get_traceback = _delegating_traceback
    return tuple(module.__name__ for module in _MODULES)


def restore_ndarray_debug_info():
    for module in _MODULES:
        original = _ORIGINAL.get(module.__name__)
        if original is not None:
            module.get_traceback = original


def active():
    return all(module.get_traceback is _delegating_traceback for module in _MODULES)
