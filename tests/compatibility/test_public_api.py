"""C01 public surface snapshots, not implementations or generated prose.

The checked-in fixture is the pre-R1 compatibility baseline. Review intentional
API changes explicitly; tests never regenerate it. Version values are excluded
because releases change them, but their exported names remain pinned.
"""

import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any, cast

import recon_graphrag as sdk
from recon_graphrag.extraction import types as extraction_types
from recon_graphrag.extraction.chunking import PageWindowBuilder, TextChunk, TextChunker
from recon_graphrag.graphdb.base import GraphWriter
from recon_graphrag.models import artifacts
from recon_graphrag.utils.tokens import TokenCounter


FIXTURES = Path(__file__).with_name("fixtures")


def public_api_snapshot():
    objects = {name: getattr(sdk, name) for name in sdk.__all__}
    # These database-neutral types are also the public writer/citation seam,
    # even though they are imported from submodules rather than the SDK root.
    for module in (extraction_types, artifacts):
        for name, obj in vars(module).items():
            if (not name.startswith("_") and inspect.isclass(obj)
                    and obj.__module__ == module.__name__ and dataclasses.is_dataclass(obj)):
                objects[f"{module.__name__}.{name}"] = obj
    for obj in (GraphWriter, TokenCounter, TextChunker, TextChunk, PageWindowBuilder):
        objects[f"{obj.__module__}.{obj.__name__}"] = obj

    surface = {}
    for name, obj in objects.items():
        if not callable(obj):
            surface[name] = {"type": type(obj).__name__}
            continue
        entry: dict[str, Any] = {
            "import_path": f"{obj.__module__}.{obj.__qualname__}",
            "signature": str(inspect.signature(obj)),
        }
        if inspect.isclass(obj):
            entry["methods"] = {
                method_name: str(inspect.signature(method))
                for method_name, method in inspect.getmembers(obj, inspect.isroutine)
                if not method_name.startswith("_")
            }
        if dataclasses.is_dataclass(obj):
            entry["fields"] = [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "default": ("<required>" if field.default is dataclasses.MISSING
                                else repr(field.default)),
                    "factory": (None if field.default_factory is dataclasses.MISSING
                                else f"{field.default_factory.__module__}.{cast(Any, field.default_factory).__qualname__}"),
                    "init": field.init,
                    "kw_only": field.kw_only,
                }
                for field in dataclasses.fields(obj)
            ]
            entry["frozen"] = cast(Any, obj).__dataclass_params__.frozen
        surface[name] = entry
    return {"exports": sdk.__all__, "surface": surface}


def test_public_exports_signatures_defaults_and_dataclass_fields():
    expected = json.loads((FIXTURES / "public_api.json").read_text())
    assert public_api_snapshot() == expected


def test_star_import_resolves_lazy_exports_and_submodule_identity():
    namespace = {}
    exec("from recon_graphrag import *", namespace)  # noqa: S102 - exercise star-import contract
    assert set(namespace) - {"__builtins__"} == set(sdk.__all__)
    for name in sdk.__all__:
        obj = getattr(sdk, name)
        assert namespace[name] is obj
        if inspect.isclass(obj) or inspect.isfunction(obj):
            module = __import__(obj.__module__, fromlist=[obj.__name__])
            assert getattr(module, obj.__name__) is obj
