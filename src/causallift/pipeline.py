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
"""Pipeline construction."""

from kedro.pipeline import Pipeline, node

from .nodes.estimate_propensity import *  # NOQA
from .nodes.model_for_each import *  # NOQA

# Here you can define your data-driven pipeline by importing your functions
# and adding them to the pipeline as follows:
#
# from nodes.data_wrangling import clean_data, compute_features
#
# pipeline = Pipeline([
#     node(clean_data, 'customers', 'prepared_customers'),
#     node(compute_features, 'prepared_customers', ['X_train', 'Y_train'])
# ])
#
# Once you have your pipeline defined, you can run it from the root of your
# project by calling:
#
# $ kedro run
#


def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """

    pipeline = Pipeline(
        [
            Pipeline(
                [
                    node(
                        bundle_train_and_test_data,
                        ["args_raw", "train_df", "test_df"],
                        "df_00",
                    )
                ],
                name="011_bundle_train_and_test_data",
            ),
            Pipeline(
                [
                    node(
                        impute_cols_features, ["args_raw", "df_00"], "args_intermediate"
                    ),
                    node(
                        schedule_propensity_scoring,
                        ["args_intermediate", "df_00"],
                        "args",
                    ),
                ],
                name="121_prepare_args",
            ),
            Pipeline(
                [
                    node(
                        treatment_fractions_,
                        ["args_raw", "df_00"],
                        "treatment_fractions",
                    )
                ],
                name="131_treatment_fractions_",
            ),
            Pipeline(
                [node(fit_propensity, ["args", "df_00"], "propensity_model")],
                name="211_fit_propensity",
            ),
            Pipeline(
                [
                    node(
                        estimate_propensity,
                        ["args", "df_00", "propensity_model"],
                        "df_01",
                    )
                ],
                name="221_estimate_propensity",
            ),
            Pipeline(
                [
                    node(
                        model_for_treated_fit, ["args", "df_01"], "treated__model_dict"
                    ),
                    node(
                        model_for_untreated_fit,
                        ["args", "df_01"],
                        "untreated__model_dict",
                    ),
                ],
                name="311_fit",
            ),
            Pipeline(
                [
                    node(
                        bundle_treated_and_untreated_models,
                        ["treated__model_dict", "untreated__model_dict"],
                        "uplift_models_dict",
                    )
                ],
                name="312_bundle_2_models",
            ),
            Pipeline(
                [
                    node(
                        model_for_treated_predict_proba,
                        ["args", "df_01", "uplift_models_dict"],
                        "treated__proba",
                    ),
                    node(
                        model_for_untreated_predict_proba,
                        ["args", "df_01", "uplift_models_dict"],
                        "untreated__proba",
                    ),
                ],
                name="321_predict_proba",
            ),
            Pipeline(
                [
                    node(
                        compute_cate,
                        ["treated__proba", "untreated__proba"],
                        "cate_estimated",
                    )
                ],
                name="411_compute_cate",
            ),
            Pipeline(
                [
                    node(
                        add_cate_to_df,
                        [
                            "args",
                            "df_01",
                            "cate_estimated",
                            "treated__proba",
                            "untreated__proba",
                        ],
                        "df_02",
                    )
                ],
                name="421_add_cate_to_df",
            ),
            Pipeline(
                [
                    node(
                        recommend_by_cate,
                        ["args", "df_02", "treatment_fractions"],
                        "df_03",
                    )
                ],
                name="511_recommend_by_cate",
            ),
            Pipeline(
                [
                    node(
                        model_for_treated_simulate_recommendation,
                        ["args", "df_03", "uplift_models_dict"],
                        "treated__sim_eval_df",
                    ),
                    node(
                        model_for_untreated_simulate_recommendation,
                        ["args", "df_03", "uplift_models_dict"],
                        "untreated__sim_eval_df",
                    ),
                ],
                name="521_simulate_recommendation",
            ),
            Pipeline(
                [
                    node(
                        estimate_effect,
                        ["treated__sim_eval_df", "untreated__sim_eval_df"],
                        "estimated_effect_df",
                    )
                ],
                name="531_estimate_effect",
            ),
            # Pipeline([
            #    node(FUNC,
            #         ['IN'],
            #         ['OUT'],
            #         ),
            # ], name='PIPELINE'),
        ]
    )

    return pipeline
