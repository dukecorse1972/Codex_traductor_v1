from __future__ import annotations

import importlib
import io
import pickle
import sys
import types
from typing import BinaryIO, Any


class MediaPipeCompatUnpickler(pickle.Unpickler):
    """Unpickler that tolerates missing `mediapipe.framework*` modules/classes.

    Some historical SWL-LSE PKLs were serialized with MediaPipe classes that lived
    under `mediapipe.framework...` paths. Newer wheels/platforms may not expose
    those modules as importable packages, causing `pickle.load` to fail with
    `ModuleNotFoundError`.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("mediapipe.framework"):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError, ImportError):
                mod = _ensure_placeholder_module(module)
                cls = getattr(mod, name, None)
                if cls is None:
                    cls = type(name, (), {})
                    setattr(mod, name, cls)
                return cls
        return super().find_class(module, name)


def _ensure_placeholder_module(module_name: str) -> types.ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception:
        pass

    # Ensure parent modules exist first (e.g., mediapipe.framework.formats)
    parts = module_name.split(".")
    for i in range(1, len(parts) + 1):
        subname = ".".join(parts[:i])
        if subname not in sys.modules:
            sys.modules[subname] = types.ModuleType(subname)
        if i > 1:
            parent_name = ".".join(parts[: i - 1])
            parent_mod = sys.modules[parent_name]
            setattr(parent_mod, parts[i - 1], sys.modules[subname])
    return sys.modules[module_name]


def mediapipe_compat_load(file_obj: BinaryIO) -> Any:
    return MediaPipeCompatUnpickler(file_obj).load()


def mediapipe_compat_load_path(path: str) -> Any:
    with open(path, "rb") as f:
        return mediapipe_compat_load(f)
