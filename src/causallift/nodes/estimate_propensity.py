""" estimate_propensity.py """

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

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

try:
    import matplotlib.pyplot as plt
except:
    print('[Warning] Could not import matplotlib.pyplot. ')


def estimate_propensity(args, df):
    if not (args.enable_ipw and (args.col_propensity not in df.columns)):
        return df

    X_train = df.xs('train')[args.cols_features]
    y_train = df.xs('train')[args.col_treatment]
    X_test = df.xs('test')[args.cols_features]
    y_test = df.xs('test')[args.col_treatment]

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

    if args.verbose >= 2:
        print('## Popensity score is estimated by logistic regression trained using:', end='')
    lr_ = linear_model.LogisticRegression(random_state=args.random_state,
                                          verbose=args.verbose)
    if args.verbose >= 2: print('')

    if args.verbose >= 3:
        print('### Parameters for grid search of Logistic regression:\n{}'.format(args.propensity_model_params))
    model = GridSearchCV(lr_, args.propensity_model_params,
                         cv=args.cv, return_train_score=False, n_jobs=-1)
    model.fit(X_train, y_train)

    if args.verbose >= 3:
        print('### Best parameter for logistic regression:\n{}'.format(model.best_params_))
    if args.verbose >= 2:
        print('\n### Coefficients of logistic regression:')
        coef_df = pd.DataFrame(model.best_estimator_.coef_.reshape(1, -1),
                               columns=args.cols_features,
                               index=['coefficient'])
        display(coef_df)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    if args.verbose >= 3:
        print('\n### Histogram of propensity score for train and test data:')
        pd.Series(proba_train).hist()
        pd.Series(proba_test).hist()
        try:
            plt.show()
        except:
            print('[Warning] Could not show the histogram.')

    # Optional evaluation and report of logistic regression
    if args.verbose >= 3:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print('\n### Score Table for logistic regression to calculate propensity score:')
        display(score_df(y_train, y_test, y_pred_train, y_pred_test))

    # if args.verbose >= 3:
        print('\n### Confusion Matrix for Train:')
        display(conf_mat_df(y_train, y_pred_train))
    # if args.verbose >= 3:
        print('\n### Confusion Matrix for Test:')
        display(conf_mat_df(y_test, y_pred_test))

    train_df = df.xs('train')
    test_df = df.xs('test')

    train_df.loc[:, args.col_propensity] = proba_train
    test_df.loc[:, args.col_propensity] = proba_test

    df = concat_train_test_df(train_df, test_df)

    return df
