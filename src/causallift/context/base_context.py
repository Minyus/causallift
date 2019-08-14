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
"""This module provides context for Kedro project."""

import abc
from typing import Any, Dict, Iterable, Optional, Union

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.runner import AbstractRunner, SequentialRunner


class BaseKedroContext(abc.ABC):
    """``KedroContext`` is the base class which holds the configuration and
    Kedro's main functionality. Project-specific context class should extend
    this abstract class and implement the all abstract methods.

    Example:
    ::

        >>> from kedro.context import KedroContext
        >>> from kedro.pipeline import Pipeline
        >>>
        >>> class ProjectContext(KedroContext):
        >>>     @property
        >>>     def pipeline(self) -> Pipeline:
        >>>         return Pipeline([])

    """

    def __init__(self):
        # type: (...) -> None
        self._catalog = DataCatalog()

    @property
    @abc.abstractmethod
    def pipeline(self):
        # type: (...) -> Pipeline
        """Abstract property for Pipeline getter.

        Returns:
            Defined pipeline.

        """
        raise NotImplementedError(
            "`{}` is a subclass of KedroContext and it must implement "
            "the `pipeline` property".format(self.__class__.__name__)
        )

    @property
    def catalog(self):
        # type: (...) -> DataCatalog
        """Read-only property referring to Kedro's ``DataCatalog`` for this context.

        Returns:
            DataCatalog defined in `catalog.yml`.

        """
        return self._catalog

    @property
    def io(self):
        # type: (...) -> DataCatalog
        """Read-only alias property referring to Kedro's ``DataCatalog`` for this
        context.

        Returns:
            DataCatalog defined in `catalog.yml`.

        """
        # pylint: disable=invalid-name
        return self._catalog

    def run(  # pylint: disable=too-many-arguments
        self,
        tags=None,  # type: Optional[Iterable[str] ]
        runner=None,  # type: Optional[AbstractRunner]
        node_names=None,  # type: Optional[Iterable[str]]
        from_nodes=None,  # type: Optional[Iterable[str]]
        to_nodes=None,  # type: Optional[Iterable[str]]
    ):
        # type: (...) -> Dict[str, Any]
        """Runs the pipeline with a specified runner.

        Args:
            tags: An optional list of node tags which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                containing *any* of these tags will be run.
            runner: An optional parameter specifying the runner that you want to run
                the pipeline with.
            node_names: An optional list of node names which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                with these names will be run.
            from_nodes: An optional list of node names which should be used as a
                starting point of the new ``Pipeline``.
            to_nodes: An optional list of node names which should be used as an
                end point of the new ``Pipeline``.
        Raises:
            KedroContextError: If the resulting ``Pipeline`` is empty
                or incorrect tags are provided.
        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.
        """

        # Load the pipeline as the intersection of all conditions
        pipeline = self.pipeline
        if tags:
            pipeline = pipeline & self.pipeline.only_nodes_with_tags(*tags)
            if not pipeline.nodes:
                raise KedroContextError(
                    "Pipeline contains no nodes with tags: {}".format(str(tags))
                )
        if from_nodes:
            pipeline = pipeline & self.pipeline.from_nodes(*from_nodes)
        if to_nodes:
            pipeline = pipeline & self.pipeline.to_nodes(*to_nodes)
        if node_names:
            pipeline = pipeline & self.pipeline.only_nodes(*node_names)

        if not pipeline.nodes:
            raise KedroContextError("Pipeline contains no nodes")

        # Run the runner
        runner = runner or SequentialRunner()
        return runner.run(pipeline, self.catalog)


class KedroContextError(Exception):
    """Error occurred when loading project and running context pipeline."""
