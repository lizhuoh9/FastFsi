"""The import graph of the fastfsi package: it imports only the standard library, NumPy, SciPy, Taichi and itself; the
fluid and the solid layers never import each other; and every module imports on the CPU runtime."""
import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED_TOP_LEVEL = set(sys.stdlib_module_names) | {"numpy", "scipy", "taichi", "fastfsi"}


def module_name(path):
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = {module_name(p): p for p in sorted((SRC / "fastfsi").rglob("*.py"))}


def imported_modules(name, path):
    """Absolute names of the modules imported anywhere in the file (function-level imports included, relative imports
    resolved, `from package import module` counted as the module)."""
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                target = ".".join(parts[:len(parts) - node.level + 1] + ([node.module] if node.module else []))
            else:
                target = node.module
            found.add(target)
            found.update(target + "." + alias.name for alias in node.names if target + "." + alias.name in MODULES)
    return found


GRAPH = {name: imported_modules(name, path) for name, path in MODULES.items()}


def test_the_package_has_modules():
    assert "fastfsi.fluid.solver" in MODULES and "fastfsi.io.restart" in MODULES and len(MODULES) > 50


def test_imports_only_the_standard_library_numpy_scipy_taichi_and_itself():
    bad = {name: sorted(m for m in imports if m.split(".")[0] not in ALLOWED_TOP_LEVEL) for name, imports in GRAPH.items()}
    assert {name: found for name, found in bad.items() if found} == {}


def test_every_imported_fastfsi_module_exists():
    missing = {name: sorted(m for m in imports if m.split(".")[0] == "fastfsi" and m not in MODULES
                            and not any(other.startswith(m + ".") for other in MODULES))
               for name, imports in GRAPH.items()}
    assert {name: found for name, found in missing.items() if found} == {}


@pytest.mark.parametrize("importer,imported", (("fastfsi.fluid", "fastfsi.solid"), ("fastfsi.solid", "fastfsi.fluid")))
def test_fluid_and_solid_never_import_each_other(importer, imported):
    bad = {name: sorted(m for m in imports if m == imported or m.startswith(imported + "."))
           for name, imports in GRAPH.items() if name == importer or name.startswith(importer + ".")}
    assert {name: found for name, found in bad.items() if found} == {}


def test_every_module_imports_on_the_cpu_runtime():
    from cpu_runtime import initialize
    initialize()
    for name in MODULES:
        importlib.import_module(name)
