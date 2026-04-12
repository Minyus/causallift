from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from causallift.context.flexible_context import (
    FlexibleKedroContext,
    ProjectContext,
    ProjectContext1,
)
from kedro.io import MemoryDataset, SharedMemoryDataCatalog
from kedro.pipeline import Pipeline, node
from kedro.runner import ParallelRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def increment(x):
    return x + 1


def double(y):
    return y * 2


def test_parallel_runner_string_maps_to_parallel_runner(monkeypatch):
    captured = {}

    def fake_super_run(self, runner=None, **kwargs):
        captured["runner_type"] = type(runner).__name__
        return runner

    monkeypatch.setattr(ProjectContext, "run", fake_super_run)

    ctx = ProjectContext1.__new__(ProjectContext1)
    ctx._catalog = SimpleNamespace()  # type: ignore[attr-defined]
    ctx._runner = None  # type: ignore[attr-defined]
    assert ProjectContext1.run(ctx, runner="ParallelRunner") is not None
    assert captured["runner_type"] == "ParallelRunner"


def test_parallel_runner_string_reuses_same_instance(monkeypatch):
    created = []

    class DummyParallelRunner:
        def __init__(self):
            created.append(self)

    DummyParallelRunner.__name__ = "ParallelRunner"

    def fake_super_run(self, runner=None, **kwargs):
        return runner

    monkeypatch.setattr("causallift.context.flexible_context.ParallelRunner", DummyParallelRunner)
    monkeypatch.setattr(ProjectContext, "run", fake_super_run)

    ctx = ProjectContext1.__new__(ProjectContext1)
    ctx._catalog = SimpleNamespace()  # type: ignore[attr-defined]
    ctx._runner = "ParallelRunner"  # type: ignore[attr-defined]

    first = ProjectContext1.run(ctx, runner="ParallelRunner")
    second = ProjectContext1.run(ctx, runner="ParallelRunner")

    assert first is second
    assert len(created) == 1


def test_parallel_runner_uses_shared_memory_catalog(monkeypatch):
    captured = {}

    def fake_parallel_run(self, pipeline, catalog, only_missing_outputs=False, **kwargs):
        captured["catalog_type"] = type(catalog).__name__
        captured["has_set_manager_datasets"] = hasattr(catalog, "set_manager_datasets")
        captured["has_validate_catalog"] = hasattr(catalog, "validate_catalog")
        return "ok"

    class DummyContext(ProjectContext):
        @property
        def pipeline(self):  # type: ignore[override]
            return SimpleNamespace(nodes=[object()])

    monkeypatch.setattr(ParallelRunner, "run", fake_parallel_run)

    ctx = DummyContext.__new__(DummyContext)
    ctx._catalog = SimpleNamespace(_datasets={"input": MemoryDataset(1)})  # type: ignore[attr-defined]
    assert DummyContext.run(ctx, runner=ParallelRunner()) == "ok"  # type: ignore[arg-type]
    assert captured == {
        "catalog_type": "SharedMemoryDataCatalog",
        "has_set_manager_datasets": True,
        "has_validate_catalog": True,
    }


def test_parallel_runner_only_wires_manager_datasets_once(monkeypatch):
    calls = []

    def fake_parallel_run(self, pipeline, catalog, only_missing_outputs=False, **kwargs):
        return "ok"

    def fake_set_manager_datasets(self, catalog):
        calls.append((self._manager, id(catalog)))

    class DummyContext(ProjectContext):
        @property
        def pipeline(self):  # type: ignore[override]
            return SimpleNamespace(nodes=[object()])

    monkeypatch.setattr(ParallelRunner, "run", fake_parallel_run)
    monkeypatch.setattr(ParallelRunner, "_set_manager_datasets", fake_set_manager_datasets)

    ctx = DummyContext.__new__(DummyContext)
    ctx._catalog = SharedMemoryDataCatalog(datasets={"input": MemoryDataset(1)})  # type: ignore[attr-defined]
    ctx._runner = None  # type: ignore[attr-defined]

    runner = ParallelRunner()
    assert DummyContext.run(ctx, runner=runner) == "ok"  # type: ignore[arg-type]
    assert DummyContext.run(ctx, runner=runner) == "ok"  # type: ignore[arg-type]

    assert len(calls) == 1


def test_only_missing_is_not_blocked_by_populated_catalog(monkeypatch):
    class DummyPipeline:
        nodes = [object()]

    class DummyRunner:
        def __init__(self):
            self.calls = []

        def _filter_pipeline_for_missing_outputs(self, pipeline, catalog):
            self.calls.append(("filter", pipeline, catalog))
            return pipeline

        def run(self, pipeline, catalog, hook_manager=None, run_id=None, only_missing_outputs=False):
            self.calls.append((pipeline, catalog, only_missing_outputs))
            return {"skipped": True}

    class DummyContext(ProjectContext):
        @property
        def pipeline(self):  # type: ignore[override]
            return DummyPipeline()

    ctx = DummyContext.__new__(DummyContext)
    ctx._catalog = type("Catalog", (), {"_datasets": {"existing": object()}})()  # type: ignore[attr-defined]

    runner = DummyRunner()
    assert DummyContext.run(ctx, runner=runner, only_missing=True) == {"skipped": True}  # type: ignore[arg-type]
    assert runner.calls[0][0].nodes
    assert runner.calls[0][2] is True


def test_project_context_preserves_from_and_to_nodes(monkeypatch):
    calls = []

    class DummyPipeline:
        nodes = [object(), object(), object()]

        def only_nodes_with_tags(self, *tags):
            calls.append(("tags", tags))
            return self

        def from_nodes(self, *nodes):
            calls.append(("from", nodes))
            return self

        def to_nodes(self, *nodes):
            calls.append(("to", nodes))
            return self

        def only_nodes(self, *nodes):
            calls.append(("nodes", nodes))
            return self

    class DummyContext(ProjectContext):
        @property
        def pipeline(self):  # type: ignore[override]
            return DummyPipeline()

    class DummyRunner:
        def run(self, pipeline, catalog, only_missing_outputs=False):
            return {"ok": True}

    ctx = DummyContext.__new__(DummyContext)
    ctx._catalog = SimpleNamespace()  # type: ignore[attr-defined]
    assert DummyContext.run(ctx, runner=DummyRunner(), from_nodes=["a"], to_nodes=["b"], node_names=["c"]) == {"ok": True}  # type: ignore[arg-type]
    assert calls == [("from", ("a",)), ("to", ("b",)), ("nodes", ("c",))]


def test_parallel_context_converts_runtime_outputs_to_shared_memory_dataset(monkeypatch):
    class DummyPipeline:
        nodes = [object()]

    class DummyRunner:
        def __init__(self):
            self.calls = []

        def run(self, pipeline, catalog, hook_manager=None, run_id=None, only_missing_outputs=False):
            self.calls.append((type(catalog).__name__, only_missing_outputs))
            return {"df_00": [1, 2, 3]}

    class DummyContext(FlexibleKedroContext):
        @property
        def pipeline(self):  # type: ignore[override]
            return DummyPipeline()

    class FakeSharedMemoryDataset:
        def __init__(self, manager=None):
            self._value = None

        def save(self, data):
            self._value = data

        def load(self):
            return self._value

    monkeypatch.setattr("causallift.context.flexible_context.SharedMemoryDataset", FakeSharedMemoryDataset)

    ctx = DummyContext.__new__(DummyContext)
    ctx._catalog = SharedMemoryDataCatalog(datasets={"seed": MemoryDataset(0)})  # type: ignore[attr-defined]
    ctx._catalog._manager = object()  # type: ignore[attr-defined]
    ctx._runner = None
    ctx._only_missing = False

    runner = DummyRunner()
    assert DummyContext.run(ctx, runner=runner) == {"df_00": [1, 2, 3]}  # type: ignore[arg-type]
    assert runner.calls == [("SharedMemoryDataCatalog", False)]
    assert type(ctx.catalog._datasets["df_00"]).__name__ == "FakeSharedMemoryDataset"


@pytest.mark.xfail(sys.platform == "darwin", reason="Kedro multiprocessing authstring issue in this environment")
def test_parallel_runner_executes_real_pipeline_end_to_end():
    script = """
from kedro.io import MemoryDataset, SharedMemoryDataCatalog
from kedro.pipeline import Pipeline, node
from kedro.runner import ParallelRunner

def increment(x):
    return x + 1

def double(y):
    return y * 2

pipeline = Pipeline([node(increment, 'x', 'y', name='inc'), node(double, 'y', 'z', name='double')])
runner = ParallelRunner()
catalog = SharedMemoryDataCatalog(datasets={'x': MemoryDataset(1)})
catalog._manager = runner._manager
runner._validate_catalog(catalog)
runner._set_manager_datasets(catalog)
runner._run(pipeline, catalog, None, 'test-run')
assert catalog.load('z') == 4
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_conditionally_skip_reruns_when_missing_outputs_are_not_cataloged(monkeypatch):
    class DummyPipeline:
        nodes = [object()]

        def outputs(self):
            return {"df_00"}

    class DummyRunner:
        def __init__(self):
            self.calls = []

        def _filter_pipeline_for_missing_outputs(self, pipeline, catalog):
            return pipeline

        def run(self, pipeline, catalog, hook_manager=None, run_id=None, only_missing_outputs=False):
            self.calls.append(only_missing_outputs)
            if only_missing_outputs:
                return {}
            return {"df_00": [1, 2, 3]}

    class DummyContext(FlexibleKedroContext):
        @property
        def pipeline(self):  # type: ignore[override]
            return DummyPipeline()

    ctx = DummyContext.__new__(DummyContext)
    ctx._catalog = SimpleNamespace(_datasets={})  # type: ignore[attr-defined]
    ctx._runner = None
    ctx._only_missing = False

    runner = DummyRunner()
    assert DummyContext.run(ctx, runner=runner, only_missing=True) == {"df_00": [1, 2, 3]}  # type: ignore[arg-type]
    assert runner.calls == [True, False]
    assert ctx.catalog._datasets["df_00"].load() == [1, 2, 3]
