# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Application entry point."""

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union, cast  # NOQA

from kedro.io import MemoryDataset, SharedMemoryDataCatalog, SharedMemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner import AbstractRunner, ParallelRunner, SequentialRunner

from causallift.context.base_context import BaseKedroContext, KedroContextError
from causallift.pipeline import create_pipeline

log = logging.getLogger(__name__)


def _as_runtime_dataset(data, shared=False, manager=None):
    if hasattr(data, "load"):
        return data
    if not shared:
        return MemoryDataset(data)
    dataset = SharedMemoryDataset(manager)
    if manager is not None:
        dataset.save(data)
    else:
        return MemoryDataset(data)
    return dataset


def _catalog_has_output(catalog, name):
    if name in getattr(catalog, "_datasets", {}):
        try:
            catalog.load(name)
            return True
        except Exception:
            return False
    if hasattr(catalog, "exists"):
        try:
            return bool(catalog.exists(name))
        except Exception:
            return False
    try:
        catalog.load(name)
        return True
    except Exception:
        return False


def _resolve_runner(runner, current=None):
    if isinstance(runner, str):
        assert runner in {"ParallelRunner", "SequentialRunner"}
        if current is not None and type(current).__name__ == runner:
            return current
        return ParallelRunner() if runner == "ParallelRunner" else SequentialRunner()
    return runner


def _needs_manager_datasets_wiring(runner, catalog):
    return (
        getattr(runner, "_manager_datasets_manager", None)
        is not getattr(runner, "_manager", None)
        or getattr(runner, "_manager_datasets_catalog", None) is not catalog
    )


class ProjectContext(BaseKedroContext):
    """Users can override the remaining methods from the parent class here, or create new ones
    (e.g. as required by plugins)

    """

    project_name = "CausalLift"
    project_version = "0.15.0"

    @property
    def pipeline(self):
        # type: (...) -> Pipeline
        return create_pipeline()

    def run(
        self,
        tags: Optional[Iterable[str]] = None,
        runner: Optional[AbstractRunner] = None,
        node_names: Optional[Iterable[str]] = None,
        from_nodes: Optional[Iterable[str]] = None,
        to_nodes: Optional[Iterable[str]] = None,
        only_missing: bool = False,
    ) -> Dict[str, Any]:
        """Runs the pipeline wi th a specified runner.

        Args:
            tags: An optional list of node tags which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                containing *any* of these tags will be run.
            runner: An optional parameter specifying the runner that you want to run
                the pipeline with.
            node_names: An optional list of node names which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                with these names will be run.
            only_missing: An option to run only missing nodes.
        Raises:
            KedroContextError: If the resulting ``Pipeline`` is empty
                or incorrect tags are provided.
        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.
        """

        # Load the pipeline
        pipeline = self.pipeline
        if tags:
            pipeline = pipeline.only_nodes_with_tags(*tags)
        if from_nodes:
            pipeline = pipeline.from_nodes(*from_nodes)
        if to_nodes:
            pipeline = pipeline.to_nodes(*to_nodes)
        if node_names:
            pipeline = pipeline.only_nodes(*node_names)

        if not pipeline.nodes:
            msg = "Pipeline contains no nodes"
            if tags:
                msg += " with tags: {}".format(str(tags))
            raise KedroContextError(msg)

        # Run the runner
        runner = runner or SequentialRunner()
        catalog = self.catalog
        if isinstance(runner, ParallelRunner):
            catalog = self._ensure_shared_memory_catalog()
            shared_catalog = cast(Any, catalog)
            setattr(catalog, "_manager", runner._manager)
            runner._validate_catalog(shared_catalog)
            if _needs_manager_datasets_wiring(runner, shared_catalog):
                runner._set_manager_datasets(shared_catalog)
                setattr(runner, "_manager_datasets_manager", runner._manager)
                setattr(runner, "_manager_datasets_catalog", shared_catalog)
        return runner.run(pipeline, catalog, only_missing_outputs=only_missing)


class ProjectContext1(ProjectContext):
    r"""Allow to specify runner by string."""

    def run(
        self,
        tags: Optional[Iterable[str]] = None,
        runner: Union[AbstractRunner, str, None] = None,
        node_names: Optional[Iterable[str]] = None,
        from_nodes: Optional[Iterable[str]] = None,
        to_nodes: Optional[Iterable[str]] = None,
        only_missing: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(getattr(self, "_runner", None), str):
            self._runner = _resolve_runner(self._runner)
        if isinstance(runner, str):
            runner = _resolve_runner(runner, getattr(self, "_runner", None))
        runner = cast(Optional[AbstractRunner], runner)
        return super().run(
            tags=tags,
            runner=runner,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            only_missing=only_missing,
        )


class ProjectContext2(ProjectContext1):
    r"""Keep the output datasets in the catalog."""

    def run(
        self,
        tags: Optional[Iterable[str]] = None,
        runner: Union[AbstractRunner, str, None] = None,
        node_names: Optional[Iterable[str]] = None,
        from_nodes: Optional[Iterable[str]] = None,
        to_nodes: Optional[Iterable[str]] = None,
        only_missing: bool = False,
    ) -> Dict[str, Any]:
        d = super().run(
            tags=tags,
            runner=runner,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            only_missing=only_missing,
        )
        shared = isinstance(self.catalog, SharedMemoryDataCatalog)
        self.catalog._datasets.update(  # type: ignore[arg-type]
            {
                key: _as_runtime_dataset(
                    value,
                    shared=shared,
                    manager=getattr(self.catalog, "_manager", None),
                )
                for key, value in d.items()
            }.items()
        )
        return d


class FlexibleKedroContext(ProjectContext2):
    r"""Overwrite the default runner and only_missing option for the run."""

    def __init__(
        self,
        runner=None,  # type: Optional[str]
        only_missing=False,  # type: bool
        **kwargs,  # type: Any
    ):
        # type: (...) -> None
        super().__init__(**kwargs)
        self._runner = _resolve_runner(runner)
        self._only_missing = only_missing

    def run(
        self,
        tags: Optional[Iterable[str]] = None,
        runner: Union[AbstractRunner, str, None] = None,
        node_names: Optional[Iterable[str]] = None,
        from_nodes: Optional[Iterable[str]] = None,
        to_nodes: Optional[Iterable[str]] = None,
        only_missing: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(getattr(self, "_runner", None), str):
            self._runner = _resolve_runner(self._runner)
        if isinstance(runner, str):
            runner = _resolve_runner(runner, getattr(self, "_runner", None))
        runner = runner or self._runner
        only_missing = only_missing or self._only_missing
        log.info(
            "Run pipeline ("
            + ("nodes: {}, ".format(node_names) if node_names else "")
            + ("tags: {}, ".format(tags) if tags else "")
            + "{}, ".format(runner)
            + "only_missing: {}".format(only_missing)
            + ")"
        )
        d = super().run(
            tags=tags,
            runner=runner,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            only_missing=only_missing,
        )

        if only_missing:
            pipeline = self.pipeline
            if tags:
                pipeline = pipeline.only_nodes_with_tags(*tags)
            if from_nodes:
                pipeline = pipeline.from_nodes(*from_nodes)
            if to_nodes:
                pipeline = pipeline.to_nodes(*to_nodes)
            if node_names:
                pipeline = pipeline.only_nodes(*node_names)
            output_names = getattr(pipeline, "outputs", lambda: set())()
            missing_outputs = [
                name
                for name in output_names
                if name not in d and not _catalog_has_output(self.catalog, name)
            ]
            if missing_outputs:
                d = super().run(
                    tags=tags,
                    runner=runner,
                    node_names=node_names,
                    from_nodes=from_nodes,
                    to_nodes=to_nodes,
                    only_missing=False,
                )

        shared = isinstance(self.catalog, SharedMemoryDataCatalog)
        self.catalog._datasets.update(  # type: ignore[arg-type]
            {
                key: _as_runtime_dataset(
                    value,
                    shared=shared,
                    manager=getattr(self.catalog, "_manager", None),
                )
                for key, value in d.items()
            }.items()
        )
        return d
