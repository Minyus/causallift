import logging

from IPython.display import display
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

from .utils import *  # NOQA

log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except:  # NOQA
    print("[Warning] Could not import matplotlib.pyplot. ")


def fit_propensity(args, df):

    X_train = df.xs("train")[args.cols_features]
    y_train = df.xs("train")[args.col_treatment]
    # X_test = df.xs("test")[args.cols_features]
    # y_test = df.xs("test")[args.col_treatment]

    # """ Transfrom by StandardScaler """
    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # """ Transform by PCA """
    # from sklearn.decomposition import PCA
    # pca = PCA(0.99)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    model = initialize_model(args, model_key="propensity_model_params")

    if args.verbose >= 2:
        log.info("## Propensity scores will be estimated by logistic regression.")

    if args.verbose >= 3:
        log.info(
            "### Parameters for grid search of Logistic regression:\n{}".format(
                args.propensity_model_params
            )
        )

    model.fit(X_train, y_train)

    best_estimator = (
        model.best_estimator_ if hasattr(model, "best_estimator_") else model
    )
    estimator_params = best_estimator.get_params()
    if "steps" in estimator_params:
        best_estimator = estimator_params["steps"][-1][1]
        estimator_params = best_estimator.get_params()

    if args.verbose >= 3:
        log.info(
            "### Best parameter for logistic regression:\n{}".format(estimator_params)
        )
    if args.verbose >= 2:
        log.info("\n## Coefficients of logistic regression:")
        coef_df = pd.DataFrame(
            best_estimator.coef_.reshape(1, -1),
            columns=args.cols_features,
            index=["coefficient"],
        )
        display(coef_df)

    return model


def estimate_propensity(args, df, model):

    X_train = df.xs("train")[args.cols_features]
    y_train = df.xs("train")[args.col_treatment]
    X_test = df.xs("test")[args.cols_features]
    y_test = df.xs("test")[args.col_treatment]

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    if args.verbose >= 3:
        log.info("\n### Histogram of propensity score for train and test data:")
        pd.Series(proba_train).hist()
        pd.Series(proba_test).hist()
        try:
            plt.show()
        except:  # NOQA
            log.info("[Warning] Could not show the histogram.")

    # Optional evaluation and report of logistic regression
    if args.verbose >= 3:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        log.info(
            "\n### Score Table for logistic regression to calculate propensity score:"
        )
        display(score_df(y_train, y_test, y_pred_train, y_pred_test))

        # if args.verbose >= 3:
        log.info("\n### Confusion Matrix for Train:")
        display(conf_mat_df(y_train, y_pred_train))
        # if args.verbose >= 3:
        log.info("\n### Confusion Matrix for Test:")
        display(conf_mat_df(y_test, y_pred_test))

    train_df = df.xs("train")
    test_df = df.xs("test")

    train_df.loc[:, args.col_propensity] = proba_train
    test_df.loc[:, args.col_propensity] = proba_test

    df = concat_train_test_df(args, train_df, test_df)

    return df


def schedule_propensity_scoring(args, df):
    args.need_propensity_scoring = args.enable_ipw and (
        args.col_propensity not in df.columns
    )
    if not args.need_propensity_scoring:
        if args.enable_ipw:
            if args.verbose >= 2:
                log.info(
                    "Skip estimation of propensity score because "
                    "{} column found in the data frame. ".format(args.col_propensity)
                )
        else:
            if args.verbose >= 2:
                log.info(
                    "Skip estimation of propensity score because "
                    '"enable_ipw" is set to False.'
                )
    return args
