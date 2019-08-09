"""
The original code is at
https://github.com/wayfair/pylift/blob/master/pylift/generate_data.py
licensed under the BSD 2-Clause "Simplified" License
Copyright 2018, Wayfair, Inc.

This code is an enhanced (backward-compatible) version that can simulate
observational dataset including "sleeping dogs."

"Sleeping dogs" (people who will "buy" if not treated but will not "buy"
if treated) can be simulated by negative values in tau parameter.
Observational data which includes confounding can be simulated by non-zero
values in propensity_coef parameter.
A/B Test (RCT) with a 50:50 split can be simulated by all-zeros values in
propensity_coef parameter (default).
The first element in each list parameter specifies the intercept.
"""

import numpy as np
import pandas as pd


def generate_data(
    N=1000,
    n_features=3,
    beta=[-3, -8, 13, -8],
    error_std=0.5,
    tau=3,
    tau_std=1,
    discrete_outcome=False,
    seed=2701,
    feature_effect=0.5,
    propensity_coef=[0, 0, 0, 0],
    index_name=None,
):
    r"""
    generate_data(N=1000, n_features=3, beta=[1,-2,3,-0.8], error_std=0.5, tau=3, discrete_outcome=False)
    Generates random data with a ground truth data generating process.
    Draws random values for features from [0, 1), errors from a 0-centered
    distribution with std `error_std`, and creates an outcome y.

    Args:
        N :
            (:obj:`Optional[int]`) -
            Number of observations.
        n_features :
            (:obj:`Optional[int]`) -
            Number of features.
        beta :
            (:obj:`Optional[List[float]]`) -
            Array of beta coefficients to multiply by X to get y.
        error_std :
            (:obj:`Optional[float]`) -
            Standard deviation (scale) of distribution from which errors are drawn.
        tau :
            (:obj:`Union[List[float], float]`) -
            Array of coefficients to multiply by X to get y if treated.
            More/larger negative values will simulate more "sleeping dogs"
            If float scalar is input, effect of features is not considered.
        tau_std :
            (:obj:`Optional[float]`) -
            When not :obj:`None`, draws tau from a normal distribution centered around tau
            with standard deviation tau_std rather than just using a constant value
            of tau.
        discrete_outcome :
            (:obj:`Optional[bool]`) -
            If True, outcomes are 0 or 1; otherwise continuous.
        seed :
             (:obj:`Optional[int]`) -
            Random seed fed to np.random.seed to allow for deterministic behavior.
        feature_effect :
            (:obj:`Optional[float]`) -
            Effect of beta on outcome if treated.
        propensity_coef :
            (:obj:`Optional[List[float]]`) -
            Array of coefficients to multiply by X to get propensity log-odds to be treated.
        index_name :
            (:obj:`Optional[str]`) -
            Index name in the output DataFrame. If :obj:`None` (default), index name will not be set.

    Returns:
        df : pd.DataFrame
            A DataFrame containing the generated data.
    """
    # Check the length of input lists
    if len(propensity_coef) != n_features + 1:
        raise Exception("len(propensity_coef) != n_features + 1")
    if len(beta) != n_features + 1:
        raise Exception("len(beta) != n_features + 1")
    if isinstance(tau, (float, int)):
        tau_ = [0] * (n_features + 1)
        tau_[0] = tau
    if isinstance(tau, (list)):
        if len(tau) != n_features + 1:
            raise Exception("len(tau) != n_features + 1")
        tau_ = tau

    np.random.seed(seed=seed)

    # Define features, error, and random treatment assignment.
    X = np.random.random(size=(N, n_features))
    error = np.random.normal(size=(N), loc=0, scale=error_std)

    # Effect of features on treatment.
    # treatment = np.random.binomial(1, .5, size=(N))
    propensity_logodds = propensity_coef[0] + np.dot(X, propensity_coef[1:])
    propensity = 1 / (1 + np.exp(-propensity_logodds))
    treatment = np.random.binomial(1, propensity, size=(N))

    # Effect of features on outcome.
    if beta is None:
        beta = np.random.random(n_features + 1)

    # Treatment heterogeneity.
    tau_vec = (
        np.random.normal(loc=tau_[0], scale=tau_std, size=N)
        + np.dot(X, tau_[1:])
        + np.dot(X, beta[1:]) * feature_effect
    )

    # Calculate outcome.
    y = beta[0] + np.dot(X, beta[1:]) + error + treatment * tau_vec

    if discrete_outcome:
        y = y > 0

    names = ["Feature_{}".format(i) for i in range(n_features)]
    names.extend(["Treatment", "Outcome"])

    df = pd.DataFrame(
        np.concatenate((X, treatment.reshape(-1, 1), y.reshape(-1, 1)), axis=1),
        columns=names,
    )

    if index_name is not None:
        df.index.name = index_name

    return df
