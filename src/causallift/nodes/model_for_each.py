""" model_for_each.py """
""" 2 supervised models """

from IPython.display import display

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
                 df_,
                 args,
                 treatment_val=1.0,
                 ):

        assert isinstance(df_, pd.DataFrame)
        assert treatment_val in [0.0, 1.0]
        seed = args.random_state
        params = args.uplift_model_params

        if args.verbose >= 2:
            print('\n\n## Model for Treatment = {}'.format(treatment_val))

        df = df_.query('{}=={}'.format(args.col_treatment, treatment_val)).copy()

        X_train = df.xs('train')[args.cols_features]
        y_train = df.xs('train')[args.col_outcome]
        X_test = df.xs('test')[args.cols_features]
        y_test = df.xs('test')[args.col_outcome]

        if args.enable_ipw and (args.col_propensity in df.xs('train').columns):
            propensity = df.xs('train')[args.col_propensity]

            # avoid propensity near 0 or 1 which will result in too large weight
            if propensity.min() < args.min_propensity and args.verbose >= 2:
                print('[Warning] Propensity scores below {} were clipped.'.format(args.min_propensity))
            if propensity.max() > args.max_propensity and args.verbose >= 2:
                print('[Warning] Propensity scores above {} were clipped.'.format(args.max_propensity))
            propensity.clip(lower=args.min_propensity, upper=args.max_propensity, inplace=True)

            sample_weight = \
                (1 / propensity) if treatment_val == 1.0 else (1 / (1 - propensity))
        else:
            # do not use sample weight
            sample_weight = np.ones_like(y_train, dtype=float)

        model = GridSearchCV(XGBClassifier(random_state=seed),
                             params, cv=args.cv, return_train_score=False, n_jobs=-1)

        model.fit(X_train, y_train, sample_weight=sample_weight)
        if args.verbose >= 3:
            print('### Best parameters of the model trained using samples with observational Treatment: {} \n {}'.
                  format(treatment_val, model.best_params_))

        if args.verbose >= 2:
            if hasattr(model.best_estimator_, 'feature_importances_'):
                fi_df = pd.DataFrame(
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
        if args.verbose >= 3:
            print('\n### Outcome estimated by the model trained using samples with observational Treatment:',
                  treatment_val)
            display(score_original_treatment_df)

        self.model = model
        self.treatment_val = treatment_val

        self.score_original_treatment_df = score_original_treatment_df

        self.treatment_fraction_train = \
            len(df_.xs('train').query('{}=={}'.format(args.col_treatment, 1.0))) / len(df_.xs('train'))
        self.treatment_fraction_test = \
            len(df_.xs('test').query('{}=={}'.format(args.col_treatment, 1.0))) / len(df_.xs('test'))
        self.df_ = df_

        self.args = args

    def predict_proba(self):
        model = self.model
        cols_features = self.args.cols_features

        df_ = self.df_

        X_train = df_.xs('train')[cols_features]
        X_test = df_.xs('test')[cols_features]

        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]

        return concat_train_test(y_pred_train, y_pred_test)

    def recommendation_by_cate(self, cate_series,
                               treatment_fraction_train=None, treatment_fraction_test=None):

        treatment_fraction_train = treatment_fraction_train or self.treatment_fraction_train
        treatment_fraction_test = treatment_fraction_test or self.treatment_fraction_test

        def recommendation(cate_series, treatment_fraction):
            rank_series = cate_series.rank(method='first', ascending=False, pct=True)
            r = np.where(rank_series <= treatment_fraction, 1.0, 0.0)
            return r

        recommendation_train = recommendation(cate_series.xs('train'), treatment_fraction_train)
        recommendation_test = recommendation(cate_series.xs('test'), treatment_fraction_test)

        self.df_.loc[:, self.args.col_recommendation] = \
            concat_train_test(recommendation_train, recommendation_test)


    def simulate_recommendation(self):
        verbose = self.args.verbose

        model = self.model
        df_ = self.df_
        treatment_val = self.treatment_val
        args = self.args
        score_original_treatment_df = self.score_original_treatment_df

        df = df_.query('{}=={}'.format(args.col_recommendation, treatment_val)).copy()

        X_train = df.xs('train')[args.cols_features]
        y_train = df.xs('train')[args.col_outcome]
        X_test = df.xs('test')[args.cols_features]
        y_test = df.xs('test')[args.col_outcome]

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
    def __init__(self, *lsargs, **kwargs):
        kwargs.update(treatment_val=1.0)
        super().__init__(*lsargs, **kwargs)


class ModelForUntreated(ModelForTreatedOrUntreated):
    def __init__(self, *lsargs, **kwargs):
        kwargs.update(treatment_val=0.0)
        super().__init__(*lsargs, **kwargs)
