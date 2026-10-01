"""Opt-in early import instrumentation, enabled only by diagnostic PYTHONPATH."""

import functools
import importlib.abc
import importlib.machinery
import os
import sys


def wrap(module, attribute, label, predicate=None):
    original = getattr(module, attribute)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        if predicate is not None and not predicate(module):
            return original(*args, **kwargs)
        from scripts.moe_diagnostic_trace import span
        with span(label):
            return original(*args, **kwargs)

    setattr(module, attribute, measured)


class DiagnosticLoader:
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if self.name == "torch.cuda":
            wrap(module, "_lazy_init", "early_cuda_lazy_initialization",
                 lambda cuda: not cuda.is_initialized())
        elif self.name == "flashinfer.jit.cpp_ext":
            wrap(module, "run_ninja", "flashinfer_ninja")
        elif self.name == "flashinfer.jit.core":
            wrap(module.JitSpec, "build_and_load", "flashinfer_build_and_load")

    def __getattr__(self, name):
        return getattr(self.original, name)


class DiagnosticFinder(importlib.abc.MetaPathFinder):
    names = {"torch.cuda", "flashinfer.jit.cpp_ext", "flashinfer.jit.core"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = DiagnosticLoader(spec.loader, fullname)
        return spec


if os.environ.get("MOE_DIAG_EARLY_IMPORTS") == "1":
    from scripts.moe_diagnostic_trace import emit
    emit("diagnostic_python_entry")
    sys.meta_path.insert(0, DiagnosticFinder())
