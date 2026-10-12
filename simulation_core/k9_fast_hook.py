"""K9 (Claude 2026-10-11; REQ: claude_k9_design_01/K9_FASTHOOK_REQ_01.md): an exactly equivalent fast path for the
attribute hook that Taichi 1.7.4's ``data_oriented`` installs on every decorated class.

Taichi's hook for a decorated class K (taichi/lang/kernel_impl.py, data_oriented._getattr) answers ``self.item`` as:
  1. method = K.__dict__.get(item); is_property = method.__class__ == property;
  2. x = method.fget if is_property else super(K, self).__getattribute__(item)  (the next decorated level, finally
     object.__getattribute__);
  3. if hasattr(x, '_is_wrapped_kernel'): kernel wrapping (and the only mutation, wrapped._is_staticmethod);
  4. return x(self) if is_property else x.
For an item that no decorated level of the instance's MRO declares as a property in its own __dict__, and whose resolved
value has no ``_is_wrapped_kernel``, every level returns object.__getattribute__(self, item) unchanged.  The fast hook
returns exactly that, and calls the saved original hook in every other case (a property at any decorated level, an
AttributeError, a kernel value), so those answers are the original's by construction.

Installation is per class, after Taichi's decorator ran, and only when: Taichi is 1.7.4; the class's hook is the pinned
1.7.4 hook (code fingerprint, one closure cell holding the class); and no class of its MRO other than decorated levels
and ``object`` defines __getattribute__.  Otherwise the class keeps Taichi's hook (counted in STATS['refused']).
``install()`` wraps taichi.lang.kernel_impl.data_oriented and taichi.data_oriented, so every class decorated afterwards
gets the fast hook; it is called by the first lines of simulation_core/__init__.py.  HIBM_K9_FAST_HOOK=0 (or
``uninstall()``) keeps or restores Taichi's hooks.  No Taichi file is edited.
"""
import atexit
import hashlib
import json
import os
import sys

PINNED_TAICHI = (1, 7, 4)
# hook_fingerprint() of taichi/lang/kernel_impl.py data_oriented._getattr (lines 1214-1237) in the installed 1.7.4.
PINNED_HOOK = 'ffb8a00c4d06ded25ab5076012d121d0edd74198ba57251a70ae80a9dada8575'
STATS = dict(installed=0, refused={})
_ORIGINALS = {}          # class -> Taichi's hook object
_STATE = dict(installed=False, decorator=None)


@atexit.register
def _k9_stats_line():
    if _STATE['installed'] or STATS['refused']:
        sys.stdout.write('K9_STATS %s\n' % json.dumps(dict(installed=STATS['installed'], refused=STATS['refused'],
                                                             active=_STATE['installed']), sort_keys=True))
        sys.stdout.flush()


def hook_fingerprint(function):
    """sha256 of the hook's bytecode, constants and names (closure contents excluded)."""
    code = function.__code__
    data = code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode() + repr(code.co_varnames).encode()
    return hashlib.sha256(data).hexdigest()


def _taichi_hook(cls, function):
    """True when `function` is Taichi 1.7.4's data_oriented hook for exactly `cls`."""
    code = getattr(function, '__code__', None)
    closure = getattr(function, '__closure__', None)
    return (code is not None and code.co_name == '_getattr' and closure is not None and len(closure) == 1
            and closure[0].cell_contents is cls and hook_fingerprint(function) == PINNED_HOOK)


def _decorated(cls):
    """The classes of cls.__mro__ that are decorated levels (Taichi's hook, or our fast hook in its place)."""
    return [k for k in cls.__mro__ if k in _ORIGINALS or _taichi_hook(k, k.__dict__.get('__getattribute__'))]


def _refuse(cls, reason):
    STATS['refused'][reason] = STATS['refused'].get(reason, 0) + 1
    return False


def _install_class(cls):
    original = cls.__dict__.get('__getattribute__')
    if cls in _ORIGINALS:
        return True
    if not _taichi_hook(cls, original):
        return _refuse(cls, 'not_the_pinned_hook')
    levels = _decorated(cls)
    for k in cls.__mro__:
        if k is not object and k not in levels and '__getattribute__' in k.__dict__:
            return _refuse(cls, 'foreign_getattribute_in_mro')
    dicts = tuple(k.__dict__ for k in levels)   # live mappingproxy views: later class edits are seen
    generic = object.__getattribute__

    def __getattribute__(self, item):
        for d in dicts:
            m = d.get(item)
            if m is not None and m.__class__ == property:
                return original(self, item)
        try:
            x = generic(self, item)
        except AttributeError:
            return original(self, item)
        if hasattr(x, '_is_wrapped_kernel'):
            return original(self, item)
        return x
    __getattribute__.k9_original = original
    __getattribute__.k9_dicts = dicts
    cls.__getattribute__ = __getattribute__
    _ORIGINALS[cls] = original
    STATS['installed'] += 1
    return True


def install():
    """Wrap Taichi's data_oriented decorator (once); the fast hook is installed on every class decorated afterwards."""
    if _STATE['installed'] or os.environ.get('HIBM_K9_FAST_HOOK', '1') == '0':
        return False
    import taichi
    from taichi.lang import kernel_impl
    if tuple(taichi.__version__) != PINNED_TAICHI:
        STATS['refused']['taichi_version'] = 1
        return False
    decorator = kernel_impl.data_oriented

    def data_oriented(cls):
        result = decorator(cls)
        _install_class(result)
        return result
    data_oriented.__doc__ = decorator.__doc__
    data_oriented.k9_decorator = decorator
    kernel_impl.data_oriented = data_oriented
    taichi.data_oriented = data_oriented
    _STATE.update(installed=True, decorator=decorator)
    return True


def uninstall():
    """Restore Taichi's hook on every patched class and Taichi's decorator."""
    for cls, original in list(_ORIGINALS.items()):
        cls.__getattribute__ = original
        del _ORIGINALS[cls]
    if _STATE['installed']:
        import taichi
        from taichi.lang import kernel_impl
        kernel_impl.data_oriented = _STATE['decorator']
        taichi.data_oriented = _STATE['decorator']
        _STATE.update(installed=False, decorator=None)
