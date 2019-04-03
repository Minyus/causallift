""" estimate_propensity.py """

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

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

try:
    import matplotlib.pyplot as plt
except:
    print('[Warning] Could not import matplotlib.pyplot. ')


def estimate_propensity(train_df,
                        test_df,
                        cols_features=None,
                        col_treatment='Treatment',
                        col_outcome='Outcome',
                        col_propensity='Propensity',
                        random_state=0,
                        verbose=2,
                        propensity_model_params=None,
                        cv=3):
    train_df = train_df.copy()
    test_df = test_df.copy()

    df = concat_train_test_df(train_df, test_df)

    if cols_features is None:
        cols_features = \
            [column for column in df.columns if column not in [col_treatment, col_outcome, col_propensity]]

    X_train = df.xs('train')[cols_features]
    y_train = df.xs('train')[col_treatment]
    X_test = df.xs('test')[cols_features]
    y_test = df.xs('test')[col_treatment]

    ## Transfrom by StandardScaler
    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(X_train)
    ##X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    ## Transform by PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(0.99)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    if verbose >= 2:
        print('## Popensity score is estimated by logistic regression trained using:', end='')
    lr_ = linear_model.LogisticRegression(random_state=random_state,
                                          verbose=verbose)
    if verbose >= 2: print('')

    if propensity_model_params is None:
        propensity_model_params = \
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

    if verbose >= 3:
        print('### Parameters for grid search of Logistic regression:\n{}'.format(propensity_model_params))
    model = GridSearchCV(lr_, propensity_model_params,
                         cv=cv, return_train_score=False, n_jobs=-1)
    model.fit(X_train, y_train)

    if verbose >= 3:
        print('### Best parameter for logistic regression:\n{}'.format(model.best_params_))
    if verbose >= 2:
        print('\n### Coefficients of logistic regression:')
        coef_df = pd.DataFrame(model.best_estimator_.coef_.reshape(1, -1),
                               columns=cols_features, index=['coefficient'])
        display(coef_df)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    if verbose >= 3:
        print('\n### Histogram of propensity score for train and test data:')
        pd.Series(proba_train).hist()
        pd.Series(proba_test).hist()
        try:
            plt.show()
        except:
            print('[Warning] Could not show the histogram.')

    # Optional evaluation and report of logistic regression
    if verbose >= 3:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print('\n### Score Table for logistic regression to calculate propensity score:')
        display(score_df(y_train, y_test, y_pred_train, y_pred_test))

    if verbose >= 3:
        print('\n### Confusion Matrix for Train:')
        display(conf_mat_df(y_train, y_pred_train))
    if verbose >= 3:
        print('\n### Confusion Matrix for Test:')
        display(conf_mat_df(y_test, y_pred_test))

    train_df.loc[:, col_propensity] = proba_train
    test_df.loc[:, col_propensity] = proba_test

    return train_df, test_df
