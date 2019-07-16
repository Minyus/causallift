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
                 cols_features=None,
                 col_treatment='Treatment',
                 col_outcome='Outcome',
                 col_propensity='Propensity',
                 col_cate='CATE',
                 col_recommendation='Recommendation',
                 min_propensity=0.01,
                 max_propensity=0.99,
                 random_state=0,
                 verbose=2,
                 uplift_model_params=None,
                 enable_ipw=True,
                 propensity_model_params=None,
                 cv=3):

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert set(train_df.columns) == set(test_df.columns)

        non_feature_cols = [col_treatment, col_outcome, col_propensity,
                            col_cate, col_recommendation]
        if cols_features is None: cols_features = \
            get_cols_features(train_df, non_feature_cols=non_feature_cols)

        train_df = train_df.copy()
        test_df = test_df.copy()

        if enable_ipw and (col_propensity not in train_df.columns):
            train_df, test_df = \
                estimate_propensity(train_df,
                                    test_df,
                                    cols_features=cols_features,
                                    col_treatment=col_treatment,
                                    col_outcome=col_outcome,
                                    col_propensity=col_propensity,
                                    random_state=random_state,
                                    verbose=verbose,
                                    propensity_model_params=propensity_model_params,
                                    cv=cv)

        self.enable_ipw = enable_ipw
        self.train_df = train_df
        self.test_df = test_df

        self.random_state = random_state
        self.verbose = verbose
        self.cols_features = cols_features
        self.col_treatment = col_treatment
        self.col_outcome = col_outcome
        self.col_propensity = col_propensity
        self.col_cate = col_cate
        self.col_recommendation = col_recommendation
        self.min_propensity = min_propensity
        self.max_propensity = max_propensity
        self.uplift_model_params = uplift_model_params
        self.cv = cv

    def estimate_cate_by_2_models(self,
                                  verbose=None):
        r"""
        Estimate CATE (Conditional Average Treatment Effect) using 2 XGBoost classifier models.
        args:
            verbose:
                How much info to show.
                If None (default), use the value set in the constructor.
        """

        train_df_ = self.train_df.copy()
        test_df_ = self.test_df.copy()
        if verbose is None: verbose = self.verbose

        model_for_treated = ModelForTreated(train_df_, test_df_,
                                            random_state=self.random_state,
                                            verbose=self.verbose,
                                            cols_features=self.cols_features,
                                            col_treatment=self.col_treatment,
                                            col_outcome=self.col_outcome,
                                            col_propensity=self.col_propensity,
                                            col_recommendation=self.col_recommendation,
                                            min_propensity=self.min_propensity,
                                            max_propensity=self.max_propensity,
                                            enable_ipw=self.enable_ipw,
                                            uplift_model_params=self.uplift_model_params,
                                            cv=self.cv)
        model_for_untreated = ModelForUntreated(train_df_, test_df_,
                                                random_state=self.random_state,
                                                verbose=self.verbose,
                                                cols_features=self.cols_features,
                                                col_treatment=self.col_treatment,
                                                col_outcome=self.col_outcome,
                                                col_propensity=self.col_propensity,
                                                col_recommendation=self.col_recommendation,
                                                min_propensity=self.min_propensity,
                                                max_propensity=self.max_propensity,
                                                enable_ipw=self.enable_ipw,
                                                uplift_model_params=self.uplift_model_params,
                                                cv=self.cv)
        self.model_for_treated = model_for_treated
        self.model_for_untreated = model_for_untreated

        if verbose >= 3:
            print('### Treatment fraction in train dataset: ',
                  model_for_treated.treatment_fraction_train)
            print('### Treatment fraction in test dataset: ',
                  model_for_treated.treatment_fraction_test)

        cate_estimated = model_for_treated.predict_proba() - model_for_untreated.predict_proba()
        self.cate_estimated = cate_estimated

        self.train_df.loc[:, self.col_cate] = cate_estimated.xs('train').values
        self.test_df.loc[:, self.col_cate] = cate_estimated.xs('test').values

        return self.train_df, self.test_df

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

        if cate_estimated is None: cate_estimated = self.cate_estimated
        if verbose is None: verbose = self.verbose

        model_for_treated = self.model_for_treated
        model_for_untreated = self.model_for_untreated

        model_for_treated.recommendation_by_cate(cate_estimated,
                                                 treatment_fraction_train=treatment_fraction_train,
                                                 treatment_fraction_test=treatment_fraction_test)
        model_for_untreated.recommendation_by_cate(cate_estimated,
                                                   treatment_fraction_train=treatment_fraction_train,
                                                   treatment_fraction_test=treatment_fraction_test)
        treated_df = model_for_treated.simulate_recommendation()
        untreated_df = model_for_untreated.simulate_recommendation()

        self.treated_df = treated_df
        self.untreated_df = untreated_df

        if self.verbose >= 3:
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
                'observed conversion rate without uplift model'] \
             + untreated_df['# samples chosen without uplift model'] * untreated_df[
                 'observed conversion rate without uplift model']) \
            / (treated_df['# samples chosen without uplift model'] + untreated_df[
                '# samples chosen without uplift model'])

        ## Recommended (with uplift model)

        estimated_effect_df['predicted conversion rate using uplift model'] = \
            (treated_df['# samples recommended by uplift model'] * treated_df[
                'predicted conversion rate using uplift model'] \
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