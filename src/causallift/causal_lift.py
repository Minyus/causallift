""" causal_lift.py """

from IPython.display import display

from .nodes.utils import *
from .nodes.model_for_each import *
from .nodes.estimate_propensity import *

import pandas as pd
import numpy as np
from easydict import EasyDict

from .parameters import parameters_
from .run import *


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
                'silent':[True],
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

    """

    def __init__(self,
                 train_df,
                 test_df,
                 **kwargs):

        self.args = parameters_()
        self.args.update(kwargs)

        assert self.args.runner in {'SequentialRunner', 'ParallelRunner', 'NoRunner'}
        # TODO # self.kedro_context = ProjectContext(Path.cwd(), env=None) if self.args.runner not in {'NoRunner'} else None
        self.kedro_context = None # TODO

        if not self.kedro_context:
            self.df = bundle_train_and_test_data(train_df, test_df)
            self.args = impute_cols_features(self.args, self.df)
            self.df = estimate_propensity(self.args, self.df)
            [self.model_treated, self.score_original_treatment_treated_df] = model_for_treated_fit(self.args, self.df)
            [self.model_untreated, self.score_original_treatment_untreated_df] = model_for_untreated_fit(self.args, self.df)
            self.treatment_fractions = treatment_fractions_(self.df, self.args.col_treatment)

        self.treatment_fraction_train = self.treatment_fractions.train # for backward compatibility
        self.treatment_fraction_test = self.treatment_fractions.test # for backward compatibility

        if self.args.verbose >= 3:
            print('### Treatment fraction in train dataset: ',
                  self.treatment_fractions.train)
            print('### Treatment fraction in test dataset: ',
                  self.treatment_fractions.test)

        self._separate_train_test()  # for backward compatibility

        self.proba_treated = None
        self.proba_untreated = None
        self.cate_estimated = None

    def _separate_train_test(self):
        self.train_df = self.df.xs('train')
        self.test_df = self.df.xs('test')
        return self.train_df, self.test_df

    def estimate_cate_by_2_models(self,
                                  verbose=None):
        r"""
        Estimate CATE (Conditional Average Treatment Effect) using 2 XGBoost classifier models.
        args:
            verbose:
                How much info to show.
                If None (default), use the value set in the constructor.
        """

        # verbose = verbose or self.args.verbose

        if not self.kedro_context:
            self.proba_treated = model_for_treated_predict_proba(self.args, self.df, self.model_treated)
            self.proba_untreated = model_for_untreated_predict_proba(self.args, self.df, self.model_untreated)
            self.cate_estimated = compute_cate(self.proba_treated, self.proba_untreated)
            self.df = add_cate_to_df(self.args, self.df, self.cate_estimated)

        return self._separate_train_test()

    def estimate_recommendation_impact(self,
                                       cate_estimated=None,
                                       treatment_fraction_train=None,
                                       treatment_fraction_test=None,
                                       verbose=None):
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
            self.cate_estimated = cate_estimated # for backward compatibility
            self.df.loc[:, self.args.col_cate] = cate_estimated.values
        self.treatment_fractions.train = treatment_fraction_train or self.treatment_fractions.train
        self.treatment_fractions.test = treatment_fraction_test or self.treatment_fractions.test

        verbose = verbose or self.args.verbose

        def recommendation_by_cate(df, args, treatment_fractions):

            cate_series = df[args.col_cate]

            def recommendation(cate_series, treatment_fraction):
                rank_series = cate_series.rank(method='first', ascending=False, pct=True)
                r = np.where(rank_series <= treatment_fraction, 1.0, 0.0)
                return r

            recommendation_train = recommendation(cate_series.xs('train'), treatment_fractions.train)
            recommendation_test = recommendation(cate_series.xs('test'), treatment_fractions.test)

            df.loc[:, args.col_recommendation] = \
                concat_train_test(recommendation_train, recommendation_test)

            return df

        df = recommendation_by_cate(self.df, self.args, self.treatment_fractions)
        self.df = df

        treated_df = model_for_treated_simulate_recommendation(self.args, self.df, self.model_treated,
                                                               self.score_original_treatment_treated_df)
        untreated_df = model_for_untreated_simulate_recommendation(self.args, self.df, self.model_untreated,
                                                                   self.score_original_treatment_untreated_df)

        self.treated_df = treated_df
        self.untreated_df = untreated_df

        if verbose >= 3:
            print('\n### Treated samples without and with uplift model:')
            display(self.treated_df)
            print('\n### Untreated samples without and with uplift model:')
            display(self.untreated_df)

        estimated_effect_df = pd.DataFrame()

        estimated_effect_df['# samples'] = \
            treated_df['# samples chosen without uplift model'] \
            + untreated_df['# samples chosen without uplift model']

        ## Original (without uplift model)

        estimated_effect_df['observed conversion rate without uplift model'] = \
            (treated_df['# samples chosen without uplift model'] * treated_df[
                'observed conversion rate without uplift model']
             + untreated_df['# samples chosen without uplift model'] * untreated_df[
                 'observed conversion rate without uplift model']) \
            / (treated_df['# samples chosen without uplift model'] + untreated_df[
                '# samples chosen without uplift model'])

        ## Recommended (with uplift model)

        estimated_effect_df['predicted conversion rate using uplift model'] = \
            (treated_df['# samples recommended by uplift model'] * treated_df[
                'predicted conversion rate using uplift model']
             + untreated_df['# samples recommended by uplift model'] * untreated_df[
                 'predicted conversion rate using uplift model']) \
            / (treated_df['# samples recommended by uplift model'] + untreated_df[
                '# samples recommended by uplift model'])

        estimated_effect_df['predicted improvement rate'] = \
            estimated_effect_df['predicted conversion rate using uplift model'] / estimated_effect_df[
                'observed conversion rate without uplift model']

        self.estimated_effect_df = estimated_effect_df
        # if verbose >= 2:
        #    print('\n## Overall (both treated and untreated) samples without and with uplift model:')
        #    display(estimated_effect_df)

        return estimated_effect_df