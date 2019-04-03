""" model_for_each.py """
""" 2 supervised models """

from .utils import (get_cols_features,
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

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class ModelForTreatedOrUntreated():
    def __init__(self,
                 train_df_,
                 test_df_,
                 treatment_val=1.0,
                 random_state=0,
                 verbose=2,
                 cols_features=None,
                 col_treatment='Treatment',
                 col_outcome='Outcome',
                 col_propensity='Propensity',
                 col_recommendation='Recommendation',
                 min_propensity=0.01,
                 max_propensity=0.99,
                 enable_ipw=True,
                 uplift_model_params=None,
                 cv=3):

        assert treatment_val in [0.0, 1.0]
        seed = random_state
        params = uplift_model_params

        if cols_features is None:
            cols_features = \
                [column for column in df.columns if column not in [ \
                    col_treatment, col_outcome, col_propensity, col_recommendation]]

        if verbose >= 2:
            print('\n\n## Model for Treatment = {}'.format(treatment_val))

        df_ = concat_train_test_df(train_df_.reset_index(drop=True).copy(),
                                   test_df_.reset_index(drop=True).copy())
        df = df_.query('{}=={}'.format(col_treatment, treatment_val)).copy()

        X_train = df.xs('train')[cols_features]
        y_train = df.xs('train')[col_outcome]
        X_test = df.xs('test')[cols_features]
        y_test = df.xs('test')[col_outcome]

        if enable_ipw and (col_propensity in df.xs('train').columns):
            propensity = df.xs('train')[col_propensity]

            # avoid propensity near 0 or 1 which will result in too large weight
            if propensity.min() < min_propensity and verbose >= 2:
                print('[Warning] Propensity scores below {} were clipped.'.format(min_propensity))
            if propensity.max() > max_propensity and verbose >= 2:
                print('[Warning] Propensity scores above {} were clipped.'.format(max_propensity))
            propensity.clip(lower=min_propensity, upper=max_propensity, inplace=True)

            sample_weight = \
                (1 / propensity) if treatment_val == 1.0 else (1 / (1 - propensity))
        else:
            # do not use sample weight
            sample_weight = 1.0

        if params is None:
            params = {
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
        model = GridSearchCV(XGBClassifier(random_state=seed),
                             params, cv=cv, return_train_score=False, n_jobs=-1)

        model.fit(X_train, y_train, sample_weight=sample_weight)
        if verbose >= 3:
            print('### Best parameters of the model trained using samples with observational Treatment: {} \n {}'.
                  format(treatment_val, model.best_params_))

        if verbose >= 2:
            if hasattr(model.best_estimator_, 'feature_importances_'):
                fi_df = pd.DataFrame( \
                    model.best_estimator_.feature_importances_.reshape(1, -1),
                    index=['feature importance'])
                print('\n### Feature importances of the model trained using samples with observational Treatment:',
                      treatment_val)
                display(fi_df)
            else:
                print('### Feature importances not available.')

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        score_original_treatment_df = score_df(y_train, y_test, y_pred_train, y_pred_test, average='binary')
        if verbose >= 3:
            print('\n### Outcome estimated by the model trained using samples with observational Treatment:',
                  treatment_val)
            display(score_original_treatment_df)

        self.model = model
        self.treatment_val = treatment_val
        self.col_treatment = col_treatment
        self.col_outcome = col_outcome
        self.cols_features = cols_features
        self.score_original_treatment_df = score_original_treatment_df

        self.treatment_fraction_train = \
            len(df_.xs('train').query('{}=={}'.format(col_treatment, 1.0))) / len(df_.xs('train'))
        self.treatment_fraction_test = \
            len(df_.xs('test').query('{}=={}'.format(col_treatment, 1.0))) / len(df_.xs('test'))
        self.df_ = df_

        self.verbose = verbose

    def predict_proba(self):
        model = self.model
        cols_features = self.cols_features

        df_ = self.df_

        X_train = df_.xs('train')[cols_features]
        X_test = df_.xs('test')[cols_features]

        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]

        return concat_train_test(y_pred_train, y_pred_test)

    def add_recommendation_column(self, recommendation_train, recommendation_test,
                                  col_recommendation='Recommendation'):
        self.df_.loc[:, col_recommendation] = \
            concat_train_test(recommendation_train, recommendation_test)
        self.col_recommendation = col_recommendation

    def recommendation_by_cate(self, cate_series,
                               treatment_fraction_train=None, treatment_fraction_test=None):
        if treatment_fraction_train is None:
            treatment_fraction_train = self.treatment_fraction_train
        if treatment_fraction_test is None:
            treatment_fraction_test = self.treatment_fraction_test

        def recommendation(cate_series, treatment_fraction):
            rank_series = cate_series.rank(method='first', ascending=False, pct=True)
            recommendation = np.where(rank_series <= treatment_fraction, 1.0, 0.0)
            return recommendation

        recommendation_train = recommendation(cate_series.xs('train'), treatment_fraction_train)
        recommendation_test = recommendation(cate_series.xs('test'), treatment_fraction_test)
        self.add_recommendation_column(recommendation_train, recommendation_test)

    def simulate_recommendation(self):
        verbose = self.verbose

        model = self.model
        df_ = self.df_
        treatment_val = self.treatment_val
        col_recommendation = self.col_recommendation
        col_outcome = self.col_outcome
        cols_features = self.cols_features
        col_outcome = self.col_outcome
        score_original_treatment_df = self.score_original_treatment_df

        df = df_.query('{}=={}'.format(col_recommendation, treatment_val)).copy()

        X_train = df.xs('train')[cols_features]
        y_train = df.xs('train')[col_outcome]
        X_test = df.xs('test')[cols_features]
        y_test = df.xs('test')[col_outcome]

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        score_recommended_treatment_df = score_df(y_train, y_test,
                                                  y_pred_train, y_pred_test, average='binary')
        if verbose >= 3:
            print('\n### Simulated outcome of samples recommended to be treatment: {} by the uplift model:'.
                  format(treatment_val))
            display(score_recommended_treatment_df)

        out_df = pd.DataFrame(index=['train', 'test'])
        out_df['# samples chosen without uplift model'] = score_original_treatment_df[['# samples']]
        out_df['observed conversion rate without uplift model'] = \
            score_original_treatment_df[['observed conversion rate']]
        out_df['# samples recommended by uplift model'] = score_recommended_treatment_df[['# samples']]
        out_df['predicted conversion rate using uplift model'] = \
            score_recommended_treatment_df[['predicted conversion rate']]

        out_df['predicted improvement rate'] = \
            out_df['predicted conversion rate using uplift model'] \
            / out_df['observed conversion rate without uplift model']

        return out_df


class ModelForTreated(ModelForTreatedOrUntreated):
    def __init__(self, *args, **kwargs):
        kwargs.update(treatment_val=1.0)
        super().__init__(*args, **kwargs)


class ModelForUntreated(ModelForTreatedOrUntreated):
    def __init__(self, *args, **kwargs):
        kwargs.update(treatment_val=0.0)
        super().__init__(*args, **kwargs)
