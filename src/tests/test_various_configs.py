from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from causallift import CausalLift, generate_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_enable_ipw_without_known_propensity():

    seed = 0

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

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


def test_enable_ipw_without_known_propensity_conditionally_skip():

    seed = 0

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
        train_df, test_df, enable_ipw=True, verbose=3, conditionally_skip=True
    )
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


def test_disable_ipw():
    seed = 0

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

    cl = CausalLift(train_df, test_df, enable_ipw=False, verbose=3)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


def test_enable_ipw_with_known_propensity():
    seed = 0

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

    test_random_propensity = True

    if test_random_propensity:
        import random

        train_df = train_df.copy()
        train_df.loc[:, "Propensity"] = [
            random.random() for _ in range(train_df.shape[0])
        ]

        test_df = test_df.copy()
        test_df.loc[:, "Propensity"] = [
            random.random() for _ in range(test_df.shape[0])
        ]

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


def test_enable_ipw_without_known_propensity_no_runner():
    seed = 0

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

    cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3, runner=None)
    train_df, test_df = cl.estimate_cate_by_2_models()
    estimated_effect_df = cl.estimate_recommendation_impact()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(estimated_effect_df, pd.DataFrame)


if __name__ == "__main__":
    test_enable_ipw_without_known_propensity()
    test_enable_ipw_without_known_propensity_conditionally_skip()
    test_disable_ipw()
    test_enable_ipw_with_known_propensity()
    test_enable_ipw_without_known_propensity_no_runner()
    # test_enable_ipw_without_known_propensity_parallel_runner()
