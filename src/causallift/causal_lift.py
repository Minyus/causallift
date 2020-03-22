from typing import List, Optional, Tuple, Type  # NOQA

from kedro.io import AbstractDataSet, CSVLocalDataSet, MemoryDataSet, PickleLocalDataSet
import numpy as np
import sklearn  # NOQA

from causallift.context.flexible_context import *  # NOQA

from .nodes.estimate_propensity import *  # NOQA
from .nodes.model_for_each import *  # NOQA

# import logging  # NOQA
# from typing import Any, Dict, List, Optional, Tuple, Union  # NOQA
#
# import pandas as pd  # NOQA
# import sklearn  # NOQA
# from easydict import EasyDict  # NOQA
# from IPython.core.display import display  # NOQA

log = logging.getLogger(__name__)


class CausalLift:
    """
    Set up datasets for uplift modeling.
    Optionally, propensity scores are estimated based on logistic regression.

    args:
        train_df:
            Pandas Data Frame containing samples used for training
        test_df:
            Pandas Data Frame containing samples used for testing
        cols_features:
            List of column names used as features.
            If :obj:`None` (default), all the columns except for outcome,
            propensity, CATE, and recommendation.
        col_treatment:
            Name of treatment column. 'Treatment' in default.
        col_outcome:
            Name of outcome column. 'Outcome' in default.
        col_propensity:
            Name of propensity column. 'Propensity' in default.
        col_cate:
            Name of CATE (Conditional Average Treatment Effect) column. 'CATE' in default.
        col_recommendation:
            Name of recommendation column. 'Recommendation' in default.
        col_weight:
            Name of weight column. 'Weight' in default.
        min_propensity:
            Minimum propensity score. 0.01 in default.
        max_propensity:
            Maximum propensity score. 0.99 in defualt.
        verbose:
            How much info to show. Valid values are:

            * :obj:`0` to show nothing
            * :obj:`1` to show only warning
            * :obj:`2` (default) to show useful info
            * :obj:`3` to show more info

        uplift_model_params:
            Parameters used to fit 2 XGBoost classifier models.

            * Optionally use `search_cv` key to specify the Search CV class name. \n
                e.g. `sklearn.model_selection.GridSearchCV` \n
                Refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            * Use `estimator` key to specify the estimator class name. \n
                e.g. `xgboost.XGBClassifier` \n
                Refer to https://xgboost.readthedocs.io/en/latest/parameter.html
            * Optionally use `const_params` key to specify the constant parameters to \
            construct the estimator.


            If :obj:`None` (default)::

                dict(
                    search_cv="sklearn.model_selection.GridSearchCV",
                    estimator="xgboost.XGBClassifier",
                    scoring=None,
                    cv=3,
                    return_train_score=False,
                    n_jobs=-1,
                    param_grid=dict(
                        max_depth=[3],
                        learning_rate=[0.1],
                        n_estimators=[100],
                        verbose=[0],
                        objective=["binary:logistic"],
                        booster=["gbtree"],
                        n_jobs=[-1],
                        nthread=[None],
                        gamma=[0],
                        min_child_weight=[1],
                        max_delta_step=[0],
                        subsample=[1],
                        colsample_bytree=[1],
                        colsample_bylevel=[1],
                        reg_alpha=[0],
                        reg_lambda=[1],
                        scale_pos_weight=[1],
                        base_score=[0.5],
                        missing=[None],
                    ),
                )

            Alternatively, estimator model object is acceptable.
            The object must have the following methods compatible with
            scikit-learn estimator interface.

                * :func:`fit`
                * :func:`predict`
                * :func:`predict_proba`

        enable_ipw:
            Enable Inverse Probability Weighting based on the estimated propensity score.
            True in default.
        enable_weighting:
            Enable Weighting.
            False in default.
        propensity_model_params:
            Parameters used to fit logistic regression model to estimate propensity score.

            * Optionally use `search_cv` key to specify the Search CV class name.\n
                e.g. `sklearn.model_selection.GridSearchCV` \n
                Refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            * Use `estimator` key to specify the estimator class name. \n
                e.g. `sklearn.linear_model.LogisticRegression` \n
                Refer to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            * Optionally use `const_params` key to specify the constant parameters \
            to construct the estimator.


            If :obj:`None` (default)::

                dict(
                    search_cv="sklearn.model_selection.GridSearchCV",
                    estimator="sklearn.linear_model.LogisticRegression",
                    scoring=None,
                    cv=3,
                    return_train_score=False,
                    n_jobs=-1,
                    param_grid=dict(
                        C=[0.1, 1, 10],
                        class_weight=[None],
                        dual=[False],
                        fit_intercept=[True],
                        intercept_scaling=[1],
                        max_iter=[100],
                        multi_class=["ovr"],
                        n_jobs=[1],
                        penalty=["l1", "l2"],
                        solver=["liblinear"],
                        tol=[0.0001],
                        warm_start=[False],
                    ),
                )

        index_name:
            Index name of the pandas data frame after resetting the index. 'index' in default. \n
            If :obj:`None`, the index will not be reset.
        partition_name:
            Additional index name to indicate the partition, train or test. 'partition' in default.
        runner:
            If set to 'SequentialRunner' (default) or 'ParallelRunner', the pipeline is run by Kedro
            sequentially or in parallel, respectively. \n
            If set to :obj:`None` , the pipeline is run by native Python. \n
            Refer to https://kedro.readthedocs.io/en/latest/04_user_guide/05_nodes_and_pipelines.html#runners
        conditionally_skip:
            *[Effective only if runner is set to either 'SequentialRunner' or 'ParallelRunner']* \n
            Skip running the pipeline if the output files already exist.
            True in default.
        dataset_catalog:
            *[Effective only if runner is set to either 'SequentialRunner' or 'ParallelRunner']* \n
            Specify dataset files to save in Dict[str, kedro.io.AbstractDataSet] format. \n
            To find available file formats, refer to https://kedro.readthedocs.io/en/latest/kedro.io.html#data-sets \n
            In default::

                dict(
                    # args_raw = CSVLocalDataSet(filepath='../data/01_raw/args_raw.csv', version=None),
                    # train_df = CSVLocalDataSet(filepath='../data/01_raw/train_df.csv', version=None),
                    # test_df = CSVLocalDataSet(filepath='../data/01_raw/test_df.csv', version=None),
                    propensity_model  = PickleLocalDataSet(
                        filepath='../data/06_models/propensity_model.pickle',
                        version=None
                    ),
                    uplift_models_dict = PickleLocalDataSet(
                        filepath='../data/06_models/uplift_models_dict.pickle',
                        version=None
                    ),
                    df_03 = CSVLocalDataSet(
                        filepath='../data/07_model_output/df.csv',
                        load_args=dict(index_col=['partition', 'index'], float_precision='high'),
                        save_args=dict(index=True, float_format='%.16e'),
                        version=None,
                    ),
                    treated__sim_eval_df = CSVLocalDataSet(
                        filepath='../data/08_reporting/treated__sim_eval_df.csv',
                        version=None,
                    ),
                    untreated__sim_eval_df = CSVLocalDataSet(
                        filepath='../data/08_reporting/untreated__sim_eval_df.csv',
                        version=None,
                    ),
                    estimated_effect_df = CSVLocalDataSet(
                        filepath='../data/08_reporting/estimated_effect_df.csv',
                        version=None,
                    ),
                )

        logging_config:
            Specify logging configuration. \n
            Refer to https://docs.python.org/3.6/library/logging.config.html#logging-config-dictschema \n
            In default::

                {'disable_existing_loggers': False,
                 'formatters': {
                     'json_formatter': {
                         'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                         'format': '[%(asctime)s|%(name)s|%(funcName)s|%(levelname)s] %(message)s',
                     },
                     'simple': {
                         'format': '[%(asctime)s|%(name)s|%(levelname)s] %(message)s',
                     },
                 },
                 'handlers': {
                     'console': {
                         'class': 'logging.StreamHandler',
                         'formatter': 'simple',
                         'level': 'INFO',
                         'stream': 'ext://sys.stdout',
                     },
                    'info_file_handler': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'simple',
                        'filename': './info.log',
                        'maxBytes': 10485760, # 10MB
                        'backupCount': 20,
                        'encoding': 'utf8',
                        'delay': True,
                    },
                     'error_file_handler': {
                         'class': 'logging.handlers.RotatingFileHandler',
                         'level': 'ERROR',
                         'formatter': 'simple',
                         'filename': './errors.log',
                         'maxBytes': 10485760,  # 10MB
                         'backupCount': 20,
                         'encoding': 'utf8',
                         'delay': True,
                     },
                 },
                 'loggers': {
                     'anyconfig': {
                         'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                         'level': 'WARNING',
                         'propagate': False,
                     },
                     'kedro.io': {
                         'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                         'level': 'WARNING',
                         'propagate': False,
                     },
                     'kedro.pipeline': {
                         'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                         'level': 'INFO',
                         'propagate': False,
                     },
                     'kedro.runner': {
                         'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                         'level': 'INFO',
                         'propagate': False,
                     },
                     'causallift': {
                         'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                         'level': 'INFO',
                         'propagate': False,
                     },
                 },
                 'root': {
                     'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                     'level': 'INFO',
                 },
                 'version': 1}

    """

    def __init__(
        self,
        train_df=None,  # type: Optional[pd.DataFrame]
        test_df=None,  # type: Optional[pd.DataFrame]
        cols_features=None,  # type: Optional[List[str]]
        col_treatment="Treatment",  # type: str
        col_outcome="Outcome",  # type: str
        col_propensity="Propensity",  # type: str
        col_proba_if_treated="Proba_if_Treated",  # type: str
        col_proba_if_untreated="Proba_if_Untreated",  # type: str
        col_cate="CATE",  # type: str
        col_recommendation="Recommendation",  # type: str
        col_weight="Weight",  # type: str
        min_propensity=0.01,  # type: float
        max_propensity=0.99,  # type: float
        verbose=2,  # type: int
        uplift_model_params=dict(
            search_cv="sklearn.model_selection.GridSearchCV",
            estimator="xgboost.XGBClassifier",
            scoring=None,
            cv=3,
            return_train_score=False,
            n_jobs=-1,
            param_grid=dict(
                random_state=[0],
                max_depth=[3],
                learning_rate=[0.1],
                n_estimators=[100],
                verbose=[0],
                objective=["binary:logistic"],
                booster=["gbtree"],
                n_jobs=[-1],
                nthread=[None],
                gamma=[0],
                min_child_weight=[1],
                max_delta_step=[0],
                subsample=[1],
                colsample_bytree=[1],
                colsample_bylevel=[1],
                reg_alpha=[0],
                reg_lambda=[1],
                scale_pos_weight=[1],
                base_score=[0.5],
                missing=[None],
            ),
        ),  # type: Union[Dict[str, List[Any]], Type[sklearn.base.BaseEstimator]]
        enable_ipw=True,  # type: bool
        enable_weighting=False,  # type: bool
        propensity_model_params=dict(
            search_cv="sklearn.model_selection.GridSearchCV",
            estimator="sklearn.linear_model.LogisticRegression",
            scoring=None,
            cv=3,
            return_train_score=False,
            n_jobs=-1,
            param_grid=dict(
                random_state=[0],
                C=[0.1, 1, 10],
                class_weight=[None],
                dual=[False],
                fit_intercept=[True],
                intercept_scaling=[1],
                max_iter=[100],
                multi_class=["ovr"],
                n_jobs=[1],
                penalty=["l1", "l2"],
                solver=["liblinear"],
                tol=[0.0001],
                warm_start=[False],
            ),
        ),  # type: Dict[str, List[Any]]
        cv=3,  # type: int
        index_name="index",  # type: str
        partition_name="partition",  # type: str
        runner="SequentialRunner",  # type: str
        conditionally_skip=False,  # type: bool
        dataset_catalog=dict(
            # args_raw = CSVLocalDataSet(filepath='../data/01_raw/args_raw.csv', version=None),
            # train_df = CSVLocalDataSet(filepath='../data/01_raw/train_df.csv', version=None),
            # test_df = CSVLocalDataSet(filepath='../data/01_raw/test_df.csv', version=None),
            propensity_model=PickleLocalDataSet(
                filepath="../data/06_models/propensity_model.pickle", version=None
            ),
            uplift_models_dict=PickleLocalDataSet(
                filepath="../data/06_models/uplift_models_dict.pickle", version=None
            ),
            df_03=CSVLocalDataSet(
                filepath="../data/07_model_output/df.csv",
                load_args=dict(
                    index_col=["partition", "index"], float_precision="high"
                ),
                save_args=dict(index=True, float_format="%.16e"),
                version=None,
            ),
            treated__sim_eval_df=CSVLocalDataSet(
                filepath="../data/08_reporting/treated__sim_eval_df.csv", version=None
            ),
            untreated__sim_eval_df=CSVLocalDataSet(
                filepath="../data/08_reporting/untreated__sim_eval_df.csv", version=None
            ),
            estimated_effect_df=CSVLocalDataSet(
                filepath="../data/08_reporting/estimated_effect_df.csv", version=None
            ),
        ),  # type: Dict[str, AbstractDataSet]
        logging_config={
            "disable_existing_loggers": False,
            "formatters": {
                "json_formatter": {
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "[%(asctime)s|%(name)s|%(funcName)s|%(levelname)s] %(message)s",
                },
                "simple": {
                    "format": "[%(asctime)s|%(name)s|%(levelname)s] %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "level": "INFO",
                    "stream": "ext://sys.stdout",
                },
                "info_file_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "filename": "./info.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 20,
                    "encoding": "utf8",
                    "delay": True,
                },
                "error_file_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "simple",
                    "filename": "./errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 20,
                    "encoding": "utf8",
                    "delay": True,
                },
            },
            "loggers": {
                "anyconfig": {
                    "handlers": ["console", "info_file_handler", "error_file_handler"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "kedro.io": {
                    "handlers": ["console", "info_file_handler", "error_file_handler"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "kedro.pipeline": {
                    "handlers": ["console", "info_file_handler", "error_file_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
                "kedro.runner": {
                    "handlers": ["console", "info_file_handler", "error_file_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
                "causallift": {
                    "handlers": ["console", "info_file_handler", "error_file_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
            "root": {
                "handlers": ["console", "info_file_handler", "error_file_handler"],
                "level": "INFO",
            },
            "version": 1,
        },  # type: Optional[Dict[str, Any]]
    ):
        # type: (...) -> None

        self.runner = None  # type: Optional[str]
        self.kedro_context = None  # type: Optional[Type[FlexibleKedroContext]]
        self.args = None  # type: Optional[Type[EasyDict]]
        self.train_df = None  # type: Optional[Type[pd.DataFrame]]
        self.test_df = None  # type: Optional[Type[pd.DataFrame]]
        self.df = None  # type: Optional[Type[pd.DataFrame]]
        self.propensity_model = None  # type: Optional[Type[sklearn.base.BaseEstimator]]
        self.uplift_models_dict = None  # type: Optional[Type[EasyDict]]
        self.treatment_fractions = None  # type: Optional[Type[EasyDict]]
        self.treatment_fraction_train = None  # type: Optional[float]
        self.treatment_fraction_test = None  # type: Optional[float]

        self.treated__proba = None  # type: Optional[Type[np.array]]
        self.untreated__proba = None  # type: Optional[Type[np.array]]
        self.cate_estimated = None  # type: Optional[Type[pd.Series]]

        self.treated__sim_eval_df = None  # type: Optional[Type[pd.DataFrame]]
        self.untreated__sim_eval_df = None  # type: Optional[Type[pd.DataFrame]]
        self.estimated_effect_df = None  # type: Optional[Type[pd.DataFrame]]

        # Instance attributes were defined above.
        if logging_config:
            logging.config.dictConfig(logging_config)

        args_raw = dict(
            cols_features=cols_features,
            col_treatment=col_treatment,
            col_outcome=col_outcome,
            col_propensity=col_propensity,
            col_proba_if_treated=col_proba_if_treated,
            col_proba_if_untreated=col_proba_if_untreated,
            col_cate=col_cate,
            col_recommendation=col_recommendation,
            col_weight=col_weight,
            min_propensity=min_propensity,
            max_propensity=max_propensity,
            verbose=verbose,
            uplift_model_params=uplift_model_params,
            enable_ipw=enable_ipw,
            enable_weighting=enable_weighting,
            propensity_model_params=propensity_model_params,
            index_name=index_name,
            partition_name=partition_name,
            runner=runner,
            conditionally_skip=conditionally_skip,
        )

        args_raw = EasyDict(args_raw)
        args_raw.update(dataset_catalog.get("args_raw", MemoryDataSet({}).load()))

        assert args_raw.runner in {"SequentialRunner", "ParallelRunner", None}
        if args_raw.runner is None and args_raw.conditionally_skip:
            log.warning(
                "[Warning] conditionally_skip option is ignored since runner is None"
            )

        self.kedro_context = FlexibleKedroContext(
            runner=args_raw.runner, only_missing=args_raw.conditionally_skip
        )

        self.runner = args_raw.runner

        if self.runner is None:
            self.df = bundle_train_and_test_data(args_raw, train_df, test_df)
            self.args = impute_cols_features(args_raw, self.df)
            self.args = schedule_propensity_scoring(self.args, self.df)
            self.treatment_fractions = treatment_fractions_(self.args, self.df)
            if self.args.need_propensity_scoring:
                self.propensity_model = fit_propensity(self.args, self.df)
                self.df = estimate_propensity(self.args, self.df, self.propensity_model)

        if self.runner:
            self.kedro_context.catalog.add_feed_dict(
                {
                    "train_df": MemoryDataSet(train_df),
                    "test_df": MemoryDataSet(test_df),
                    "args_raw": MemoryDataSet(args_raw),
                },
                replace=True,
            )
            self.kedro_context.catalog.add_feed_dict(dataset_catalog, replace=True)

            self.kedro_context.run(tags=["011_bundle_train_and_test_data"])
            self.df = self.kedro_context.catalog.load("df_00")

            self.kedro_context.run(
                tags=[
                    "121_prepare_args",
                    "131_treatment_fractions_",
                    "141_initialize_model",
                ]
            )
            self.args = self.kedro_context.catalog.load("args")
            self.treatment_fractions = self.kedro_context.catalog.load(
                "treatment_fractions"
            )

            if self.args.need_propensity_scoring:
                self.kedro_context.run(tags=["211_fit_propensity"])
                self.propensity_model = self.kedro_context.catalog.load(
                    "propensity_model"
                )
                self.kedro_context.run(tags=["221_estimate_propensity"])
                self.df = self.kedro_context.catalog.load("df_01")
            else:
                self.kedro_context.catalog.add_feed_dict(
                    {"df_01": MemoryDataSet(self.df)}, replace=True
                )

        self.treatment_fraction_train = self.treatment_fractions.train
        self.treatment_fraction_test = self.treatment_fractions.test

        if self.args.verbose >= 3:
            log.info(
                "### Treatment fraction in train dataset: {}".format(
                    self.treatment_fractions.train
                )
            )
            log.info(
                "### Treatment fraction in test dataset: {}".format(
                    self.treatment_fractions.test
                )
            )

        self._separate_train_test()

    def _separate_train_test(self):
        # type: (...) -> Tuple[pd.DataFrame, pd.DataFrame]

        self.train_df = self.df.xs("train")
        self.test_df = self.df.xs("test")
        return self.train_df, self.test_df

    def estimate_cate_by_2_models(self):
        # type: (...) -> Tuple[pd.DataFrame, pd.DataFrame]
        r"""
        Estimate CATE (Conditional Average Treatment Effect) using 2 XGBoost classifier models.

        """

        if self.runner is None:
            treated__model_dict = model_for_treated_fit(self.args, self.df)
            untreated__model_dict = model_for_untreated_fit(self.args, self.df)
            self.uplift_models_dict = bundle_treated_and_untreated_models(
                treated__model_dict, untreated__model_dict
            )

            self.treated__proba = model_for_treated_predict_proba(
                self.args, self.df, self.uplift_models_dict
            )
            self.untreated__proba = model_for_untreated_predict_proba(
                self.args, self.df, self.uplift_models_dict
            )
            self.cate_estimated = compute_cate(
                self.treated__proba, self.untreated__proba
            )
            self.df = add_cate_to_df(
                self.args,
                self.df,
                self.cate_estimated,
                self.treated__proba,
                self.untreated__proba,
            )

        if self.runner:
            self.kedro_context.run(tags=["311_fit", "312_bundle_2_models"])
            self.uplift_models_dict = self.kedro_context.catalog.load(
                "uplift_models_dict"
            )

            self.kedro_context.run(tags=["321_predict_proba"])
            self.treated__proba = self.kedro_context.catalog.load("treated__proba")
            self.untreated__proba = self.kedro_context.catalog.load("untreated__proba")
            self.kedro_context.run(tags=["411_compute_cate"])
            self.cate_estimated = self.kedro_context.catalog.load("cate_estimated")
            self.kedro_context.run(tags=["421_add_cate_to_df"])
            self.df = self.kedro_context.catalog.load("df_02")

        return self._separate_train_test()

    def estimate_recommendation_impact(
        self,
        cate_estimated=None,  # type: Optional[Type[pd.Series]]
        treatment_fraction_train=None,  # type: Optional[float]
        treatment_fraction_test=None,  # type: Optional[float]
        verbose=None,  # type: Optional[int]
    ):
        # type: (...) -> Type[pd.DataFrame]
        r"""
        Estimate the impact of recommendation based on uplift modeling.

        args:
            cate_estimated:
                Pandas series containing the CATE.
                If :obj:`None` (default), use the ones calculated by estimate_cate_by_2_models method.
            treatment_fraction_train:
                The fraction of treatment in train dataset.
                If :obj:`None` (default), use the ones calculated by estimate_cate_by_2_models method.
            treatment_fraction_test:
                The fraction of treatment in test dataset.
                If :obj:`None` (default), use the ones calculated by estimate_cate_by_2_models method.
            verbose:
                How much info to show.
                If :obj:`None` (default), use the value set in the constructor.
        """

        if cate_estimated is not None:
            self.cate_estimated = cate_estimated
            self.df.loc[:, self.args.col_cate] = cate_estimated.values
        self.treatment_fractions.train = (
            treatment_fraction_train or self.treatment_fractions.train
        )
        self.treatment_fractions.test = (
            treatment_fraction_test or self.treatment_fractions.test
        )

        verbose = verbose or self.args.verbose

        if self.runner is None:
            self.df = recommend_by_cate(self.args, self.df, self.treatment_fractions)
            self.treated__sim_eval_df = model_for_treated_simulate_recommendation(
                self.args, self.df, self.uplift_models_dict
            )
            self.untreated__sim_eval_df = model_for_untreated_simulate_recommendation(
                self.args, self.df, self.uplift_models_dict
            )
            self.estimated_effect_df = estimate_effect(
                self.treated__sim_eval_df, self.untreated__sim_eval_df
            )

        if self.runner:
            # self.kedro_context.catalog.save('args', self.args)
            self.kedro_context.run(tags=["511_recommend_by_cate"])
            self.df = self.kedro_context.catalog.load("df_03")

            self.kedro_context.run(tags=["521_simulate_recommendation"])
            self.treated__sim_eval_df = self.kedro_context.catalog.load(
                "treated__sim_eval_df"
            )
            self.untreated__sim_eval_df = self.kedro_context.catalog.load(
                "untreated__sim_eval_df"
            )

            self.kedro_context.run(tags=["531_estimate_effect"])
            self.estimated_effect_df = self.kedro_context.catalog.load(
                "estimated_effect_df"
            )

        if verbose >= 3:
            log.info("\n### Treated samples without and with uplift model:")
            display(self.treated__sim_eval_df)
            log.info("\n### Untreated samples without and with uplift model:")
            display(self.untreated__sim_eval_df)

        # if verbose >= 2:
        #    log.info('\n## Overall (both treated and untreated) samples without and with uplift model:')
        #    display(estimated_effect_df)

        return self.estimated_effect_df
