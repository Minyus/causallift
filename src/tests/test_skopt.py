from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from causallift import CausalLift, generate_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_skopt():

    seed = 0

    uplift_model_params = dict(
        booster=Categorical(["gbtree"]),
        silent=Categorical([True]),
        objective=Categorical(["binary:logistic"]),
        base_score=Categorical([0.5]),
        eval_metric=Categorical(["auc"]),
        n_jobs=Categorical([-1]),
        seed=Categorical([seed]),
        n_estimators=Integer(100, 200),
    )
    estimator = XGBClassifier()

    model = BayesSearchCV(
        estimator=estimator,
        search_spaces=uplift_model_params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        n_iter=10,
        verbose=1,
        refit=True,
    )

    df = generate_data(
        N=1000,
        n_features=3,
        beta=[0, -2, 3, -5],  # Effect of [intercept and features] on outcome
        error_std=0.1,
        tau=[1, -5, -5, 10],  # Effect of [intercept and features] on treated outcome
        tau_std=0.1,
        discrete_outcome=True,
        seed=seed,
        feature_effect=0,  # Effect of beta on treated outxome
        propensity_coef=[
            0,
            -1,
            1,
            -1,
        ],  # Effect of [intercept and features] on propensity log-odds for treatment
        index_name="index",
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["Treatment"]
    )

    cl = CausalLift(
        train_df, test_df, enable_ipw=True, verbose=3, uplift_model_params=model
    )
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


if __name__ == "__main__":
    test_skopt()
