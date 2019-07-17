""" causal_lift.py """

from IPython.display import display

from .nodes.utils import (get_cols_features,
                    concat_train_test,
                    concat_train_test_df,
                    len_t,
                    len_o,
                    len_to,
                    treatment_fraction_,
                    outcome_fraction_,
                    overall_uplift_gain_,
                    gain_tuple,
                    score_df,
                    conf_mat_df)
from .nodes.model_for_each import (ModelForTreatedOrUntreated,
                             ModelForTreated,
                             ModelForUntreated)
from .nodes.estimate_propensity import estimate_propensity

import pandas as pd
import numpy as np
from easydict import EasyDict


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

        args = EasyDict()
        args.cols_features = None
        args.col_treatment = 'Treatment'
        args.col_outcome = 'Outcome'
        args.col_propensity = 'Propensity'
        args.col_cate = 'CATE'
        args.col_recommendation = 'Recommendation'
        args.min_propensity = 0.01
        args.max_propensity = 0.99
        args.random_state = 0
        args.verbose = 2
        args.uplift_model_params = {
            'max_depth': [3],
            'learning_rate': [0.1],
            'n_estimators': [100],
            'silent': [True],
            'objective': ['binary:logistic'],
            'booster': ['gbtree'],
            'n_jobs': [-1],
            'nthread': [None],
            'gamma': [0],
            'min_child_weight': [1],
            'max_delta_step': [0],
            'subsample': [1],
            'colsample_bytree': [1],
            'colsample_bylevel': [1],
            'reg_alpha': [0],
            'reg_lambda': [1],
            'scale_pos_weight': [1],
            'base_score': [0.5],
            'missing': [None],
        }
        args.enable_ipw = True
        args.propensity_model_params = \
            {
                'C': [0.1, 1, 10],
                'class_weight': [None],
                'dual': [False],
                'fit_intercept': [True],
                'intercept_scaling': [1],
                'max_iter': [100],
                'multi_class': ['ovr'],
                'n_jobs': [1],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'tol': [0.0001],
                'warm_start': [False]
            }
        args.cv = 3
        args.update(kwargs)

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert set(train_df.columns) == set(test_df.columns)

        non_feature_cols = [
            args.col_treatment,
            args.col_outcome,
            args.col_propensity,
            args.col_cate,
            args.col_recommendation,
        ]

        args.cols_features = \
            args.cols_features or get_cols_features(train_df, non_feature_cols=non_feature_cols)

        train_df = train_df.reset_index(drop=True).copy()
        test_df = test_df.reset_index(drop=True).copy()

        if args.enable_ipw and (args.col_propensity not in train_df.columns):
            train_df, test_df = estimate_propensity(train_df, test_df, args)

        self.df = concat_train_test_df(train_df, test_df)
        self.separate_train_test() # for backward compatibility
        self.args = args

    def separate_train_test(self):
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

        verbose = verbose or self.args.verbose

        model_for_treated = ModelForTreated(self.df, self.args)
        model_for_untreated = ModelForUntreated(self.df, self.args)
        self.model_for_treated = model_for_treated
        self.model_for_untreated = model_for_untreated

        df = self.df
        args = self.args
        self.treatment_fraction_train = \
            len(df.xs('train').query('{}=={}'.format(args.col_treatment, 1.0))) / len(df.xs('train'))
        self.treatment_fraction_test = \
            len(df.xs('test').query('{}=={}'.format(args.col_treatment, 1.0))) / len(df.xs('test'))

        if verbose >= 3:
            print('### Treatment fraction in train dataset: ',
                  self.treatment_fraction_train)
            print('### Treatment fraction in test dataset: ',
                  self.treatment_fraction_test)

        cate_estimated = model_for_treated.predict_proba(self.df) - model_for_untreated.predict_proba(self.df)
        self.cate_estimated = cate_estimated # for backward compatibility
        self.df.loc[:, self.args.col_cate] = cate_estimated.values

        return self.separate_train_test()

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
        treatment_fraction_train = treatment_fraction_train or self.treatment_fraction_train
        treatment_fraction_test = treatment_fraction_test or self.treatment_fraction_test

        verbose = verbose or self.args.verbose

        model_for_treated = self.model_for_treated
        model_for_untreated = self.model_for_untreated

        # model_for_treated.recommendation_by_cate(self.df,
        #                                          treatment_fraction_train=treatment_fraction_train,
        #                                          treatment_fraction_test=treatment_fraction_test)
        # model_for_untreated.recommendation_by_cate(self.df,
        #                                            treatment_fraction_train=treatment_fraction_train,
        #                                            treatment_fraction_test=treatment_fraction_test)
        def recommendation_by_cate(self, df_,
                                   treatment_fraction_train=None, treatment_fraction_test=None):

            def recommendation(cate_series, treatment_fraction):
                rank_series = cate_series.rank(method='first', ascending=False, pct=True)
                r = np.where(rank_series <= treatment_fraction, 1.0, 0.0)
                return r

            cate_series = df_[self.args.col_cate]
            recommendation_train = recommendation(cate_series.xs('train'), treatment_fraction_train)
            recommendation_test = recommendation(cate_series.xs('test'), treatment_fraction_test)

            df_.loc[:, self.args.col_recommendation] = \
                concat_train_test(recommendation_train, recommendation_test)
            # TODO Refactor

            return df_

        df = recommendation_by_cate(self, self.df,
                                    treatment_fraction_train=treatment_fraction_train,
                                    treatment_fraction_test=treatment_fraction_test)

        treated_df = model_for_treated.simulate_recommendation(df)
        untreated_df = model_for_untreated.simulate_recommendation(df)

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