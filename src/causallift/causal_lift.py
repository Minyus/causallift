""" causal_lift.py """

from .nodes.model_for_each import *
from .nodes.estimate_propensity import *

from .default.parameters import *
from .default.catalog import *
from .run import *

from easydict import EasyDict
from kedro.io import MemoryDataSet, AbstractDataSet

import logging
log = logging.getLogger(__name__)


class CausalLift():
    r"""
    Set up datasets for uplift modeling.
    Optionally, propensity scores are estimated based on logistic regression.

    args:
        train_df: pd.DataFrame
            Pandas Data Frame containing samples used for training
        test_df: pd.DataFrame
            Pandas Data Frame containing samples used for testing
        cols_features: list of str, optional
            List of column names used as features.
            If None (default), all the columns except for outcome,
            propensity, CATE, and recommendation.
        col_treatment: str, optional
            Name of treatment column. 'Treatment' in default.
        col_outcome: str, optional
            Name of outcome column. 'Outcome' in default.
        col_propensity: str, optional
            Name of propensity column. 'Propensity' in default.
        col_cate: str, optional
            Name of CATE (Conditional Average Treatment Effect) column. 'CATE' in default.
        col_recommendation: str, optional
            Name of recommendation column. 'Recommendation' in default.
        min_propensity: float, optional
            Minimum propensity score. 0.01 in default.
        max_propensity: float, optional
            Maximum propensity score. 0.99 in defualt.
        random_state: int, optional
            The seed used by the random number generator. 0 in default.
        verbose: int, optional
            How much info to show. Valid values are:
            0 to show nothing,
            1 to show only warning,
            2 (default) to show useful info,
            3 to show more info.
        uplift_model_params: dict, optional
            Parameters used to fit 2 XGBoost classifier models.
            Refer to https://xgboost.readthedocs.io/en/latest/parameter.html
            If None (default):
                {
                'max_depth':[3],
                'learning_rate':[0.1],
                'n_estimators':[100],
                'verbose':[0],
                'objective':['binary:logistic'],
                'booster':['gbtree'],
                'n_jobs':[-1],
                'nthread':[None],
                'gamma':[0],
                'min_child_weight':[1],
                'max_delta_step':[0],
                'subsample':[1],
                'colsample_bytree':[1],
                'colsample_bylevel':[1],
                'reg_alpha':[0],
                'reg_lambda':[1],
                'scale_pos_weight':[1],
                'base_score':[0.5],
                'missing':[None],
                }
        enable_ipw: boolean, optional
            Enable Inverse Probability Weighting based on the estimated propensity score.
            True in default.
        propensity_model_params: dict, optional
            Parameters used to fit logistic regression model to estimate propensity score.
            Refer to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            If None (default):
                {
                'C': [0.1, 1, 10],
                'class_weight': [None],
                'dual': [False],
                'fit_intercept': [True],
                'intercept_scaling': [1],
                'max_iter': [100],
                'multi_class': ['ovr'],
                'n_jobs': [1],
                'penalty': ['l1','l2'],
                'solver': ['liblinear'],
                'tol': [0.0001],
                'warm_start': [False]
                }
        cv: int, optional
            Cross-Validation for the Grid Search. 3 in default.
            Refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        runner: str, optional
            If set to 'SequentialRunner' (default) or 'ParallelRunner', the pipeline is run by kedro
            sequentially or in parallel, respectively.
            If set to None, the pipeline is run by native Python.
        conditionally_skip: bool, optional
            [Effective only if runner is set to either 'SequentialRunner' or 'ParallelRunner']
            Skip running the pipeline if the output files already exist.
            True in default.
        dataset_catalog: dict, optional
            [Effective only if runner is set to either 'SequentialRunner' or 'ParallelRunner']
            Specify dataset files to save in Dict[str, kedro.io.AbstractDataSet] format.
            To find available file formats, refer to https://kedro.readthedocs.io/en/latest/kedro.io.html#data-sets
            In default:
                dict(
                    propensity_model  = PickleLocalDataSet(
                        filepath='../data/06_models/propensity_model.pickle',
                        version=None
                    ),
                    models_dict = PickleLocalDataSet(
                        filepath='../data/06_models/models_dict.pickle',
                        version=None
                    ),
                    )
        logging_config: dict, optional
            Specify logging configuration.
            Refer to https://docs.python.org/3.6/library/logging.config.html#logging-config-dictschema
            In defalut:
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

    def __init__(self,
                 train_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 dataset_catalog: Dict[str, AbstractDataSet] = {},
                 logging_config: Dict[str, Any] = {},
                 **kwargs):

        self.runner = None
        self.kedro_context = None
        self.args = None
        self.train_df = None
        self.test_df = None
        self.df = None
        self.propensity_model = None
        self.models_dict = None
        self.treatment_fractions = None
        self.treatment_fraction_train = None
        self.treatment_fraction_test = None

        self.treated__proba = None
        self.untreated__proba = None
        self.cate_estimated = None

        self.treated__sim_eval_df = None
        self.untreated__sim_eval_df= None
        self.estimated_effect_df = None

        # Instance attributes were defined above.

        args_raw = EasyDict(parameters_())
        args_raw.update(kwargs)
        args_raw.update(dataset_catalog.get('args_raw', MemoryDataSet({}).load()))

        assert args_raw.runner in {'SequentialRunner', 'ParallelRunner', None}
        if args_raw.runner is None and args_raw.conditionally_skip:
            log.warning('[Warning] conditionally_skip option is ignored since runner is None')

        self.kedro_context = FlexibleProjectContext(
            logging_config=logging_config,
            runner=args_raw.runner,
            only_missing=args_raw.conditionally_skip,
        )

        self.runner = args_raw.runner

        if self.runner is None:
            self.df = bundle_train_and_test_data(train_df, test_df)
            self.args = impute_cols_features(args_raw, self.df)
            self.treatment_fractions = treatment_fractions_(self.args, self.df)

            self.propensity_model = fit_propensity(self.args, self.df)
            self.df = estimate_propensity(self.args, self.df, self.propensity_model)

        if self.runner:
            self.kedro_context.catalog.add_feed_dict({
                'train_df': MemoryDataSet(train_df),
                'test_df': MemoryDataSet(test_df),
                'args_raw': MemoryDataSet(args_raw),
            }, replace=True)
            self.kedro_context.catalog.add_feed_dict(datasets_(), replace=True)
            self.kedro_context.catalog.add_feed_dict(dataset_catalog, replace=True)

            self.kedro_context.run(tags=[
                '011_bundle_train_and_test_data',
                ])
            self.df = self.kedro_context.catalog.load('df_00')

            self.kedro_context.run(tags=[
                '121_impute_cols_features',
                '131_treatment_fractions_',
                ])
            self.args = self.kedro_context.catalog.load('args')
            self.treatment_fractions = self.kedro_context.catalog.load(
                'treatment_fractions')

            self.kedro_context.run(tags=[
                '211_fit_propensity',
                ])
            self.propensity_model = self.kedro_context.catalog.load('propensity_model')
            self.kedro_context.run(tags=[
                '221_estimate_propensity',
                ])
            self.df = self.kedro_context.catalog.load('df_01')

        self.treatment_fraction_train = self.treatment_fractions.train
        self.treatment_fraction_test = self.treatment_fractions.test

        if self.args.verbose >= 3:
            log.info('### Treatment fraction in train dataset: {}'.
                     format(self.treatment_fractions.train))
            log.info('### Treatment fraction in test dataset: {}'.
                     format(self.treatment_fractions.test))

        self._separate_train_test()

    def _separate_train_test(self):
        self.train_df = self.df.xs('train')
        self.test_df = self.df.xs('test')
        return self.train_df, self.test_df

    def estimate_cate_by_2_models(self):
        r"""
        Estimate CATE (Conditional Average Treatment Effect) using 2 XGBoost classifier models.
        args:

        """

        if self.runner is None:
            treated__model_dict = model_for_treated_fit(self.args, self.df)
            untreated__model_dict = model_for_untreated_fit(self.args, self.df)
            self.models_dict = bundle_treated_and_untreated_models(
                treated__model_dict, untreated__model_dict)

            self.treated__proba = model_for_treated_predict_proba(
                self.args, self.df, self.models_dict)
            self.untreated__proba = model_for_untreated_predict_proba(
                self.args, self.df, self.models_dict)
            self.cate_estimated = compute_cate(self.treated__proba, self.untreated__proba)
            self.df = add_cate_to_df(self.args, self.df, self.cate_estimated)

        if self.runner:
            self.kedro_context.run(tags=[
                '311_fit',
                '312_bundle_2_models',
                ])
            self.models_dict = self.kedro_context.catalog.load('models_dict')

            self.kedro_context.run(tags=[
                '321_predict_proba',
            ])
            self.treated__proba = self.kedro_context.catalog.load('treated__proba')
            self.untreated__proba = self.kedro_context.catalog.load('untreated__proba')
            self.kedro_context.run(tags=[
                '411_compute_cate',
            ])
            self.cate_estimated = self.kedro_context.catalog.load('cate_estimated')
            self.kedro_context.run(tags=[
                '421_add_cate_to_df'
            ])
            self.df = self.kedro_context.catalog.load('df_02')

        return self._separate_train_test()

    def estimate_recommendation_impact(self,
                                       cate_estimated: pd.Series = None,
                                       treatment_fraction_train: float = None,
                                       treatment_fraction_test: float = None,
                                       verbose: int = None):
        r"""
        Estimate the impact of recommendation based on uplift modeling.
        args:
            cate_estimated:
                Pandas series containing the CATE.
                If None (default), use the ones calculated by estimate_cate_by_2_models method.
            treatment_fraction_train:
                The fraction of treatment in train dataset.
                If None (default), use the ones calculated by estimate_cate_by_2_models method.
            treatment_fraction_test:
                The fraction of treatment in test dataset.
                If None (default), use the ones calculated by estimate_cate_by_2_models method.
            verbose:
                How much info to show.
                If None (default), use the value set in the constructor.
        """

        if cate_estimated is not None:
            self.cate_estimated = cate_estimated
            self.df.loc[:, self.args.col_cate] = cate_estimated.values
        self.treatment_fractions.train = treatment_fraction_train or self.treatment_fractions.train
        self.treatment_fractions.test = treatment_fraction_test or self.treatment_fractions.test

        verbose = verbose or self.args.verbose

        if self.runner is None:
            self.df = recommend_by_cate(self.args, self.df, self.treatment_fractions)
            self.treated__sim_eval_df = model_for_treated_simulate_recommendation(
                self.args, self.df, self.models_dict)
            self.untreated__sim_eval_df = model_for_untreated_simulate_recommendation(
                self.args, self.df, self.models_dict)
            self.estimated_effect_df = estimate_effect(
                self.treated__sim_eval_df, self.untreated__sim_eval_df)

        if self.runner:
            # self.kedro_context.catalog.save('args', self.args)
            self.kedro_context.run(tags=[
                '511_recommend_by_cate',
            ])
            self.df = self.kedro_context.catalog.load('df_03')

            self.kedro_context.run(tags=[
                '521_simulate_recommendation',
            ])
            self.treated__sim_eval_df = self.kedro_context.catalog.load('treated__sim_eval_df')
            self.untreated__sim_eval_df = self.kedro_context.catalog.load('untreated__sim_eval_df')

            self.kedro_context.run(tags=[
                '531_estimate_effect'
            ])
            self.estimated_effect_df = self.kedro_context.catalog.load('estimated_effect_df')

        if verbose >= 3:
            log.info('\n### Treated samples without and with uplift model:')
            display(self.treated__sim_eval_df)
            log.info('\n### Untreated samples without and with uplift model:')
            display(self.untreated__sim_eval_df)

        # if verbose >= 2:
        #    log.info('\n## Overall (both treated and untreated) samples without and with uplift model:')
        #    display(estimated_effect_df)

        return self.estimated_effect_df