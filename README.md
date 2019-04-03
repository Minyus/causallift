# CausalLift: Python package for Uplift Modeling for A/B testing and observational data

[![PyPI version](https://badge.fury.io/py/causallift.svg)](https://badge.fury.io/py/causallift)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD-yellow.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minyus/causallift/blob/master/examples/CausalLift_with_simulated_observational_data.ipynb)


### What kind of data can be used?
Table data including the following columns:

- features (a.k.a independent variable, explanatory variable)
- outcome (a.k.a dependent variable, target variable, label)
- treatment (a variable that can be controlled for each target, e.g. giving a drug to patients, executing an advertising campaign to customers, etc.)
- propensity (optional; CausalLift can calculate from observational data if not provided. Not needed for A/B Testing data.)

### How to install CausalLift?

Option 1: install from the PyPI

```bash
	pip3 install causallift
```

Option 2: install from the GitHub repository

```bash
	pip3 install git+https://github.com/Minyus/causallift.git
```

Option 3: clone the [GitHub repository](https://github.com/Minyus/causallift.git), cd into the downloaded repository, and run:

```bash
	python setup.py install
```

### Dependencies:

- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost


### How to use CausalLift?

Please see the [CausalLift example]( 
https://colab.research.google.com/github/Minyus/causallift/blob/master/examples/CausalLift_with_simulated_observational_data.ipynb
) available in Google Colab (free cloud CPU/GPU) environment.

To run the code, navigate to "Runtime" >> "Run all".

To download the notebook file, navigate to "File" >> "Download .ipynb".

Here are the basic steps to use.

```python
""" Step 0. Import CausalLift """

from causallift import CausalLift

""" Step 1. Instantiate CausalLift class
Feed pandas dataframes for train and test. 
If your data is observational data (not A/B Testing or Randomized Controlled Trial) and you can assume the propensity to be treated can be estimated by the features, set enable_ipw = True to use Inverse Probability Weighting.
If the fed dataframes include propensity column, CausalLift will use it.
Otherwise, CausalLift will estimate propensity using logistic regression.
"""

cl = CausalLift(train_df, test_df, enable_ipw=True)


""" Step 2. Train 2 supervised classification models (currently only XGBoost is supported) for treated and untreated samples independently and estimate CATE (Conditional Average Treatment Effect). It is recommended to treat only the CATE is high enough """

train_df, test_df = cl.estimate_cate_by_2_models()


""" Step 3. Estimate the impact of choosing treatment targets as recommended by the uplift modeling.
"""
estimated_effect_df = cl.estimate_recommendation_impact()
```

### Why CausalLift was developed?

- Existing packages for Uplift Modeling assumes the dataset is A/B Testing (Randomized Control Trial) dataset. In real business, however, observational dataset (dataset in which treatment was not chosen randomly) is more common. CausalLift utilizes basic methodology in Causal Inference ("Inverse Probability Weighting") and Uplift Modeling (training 2 models independently for treated and untreated samples.).
	
- Metrics used to evaluate Uplift Modeling such as Qini are useful in research, but difficult to use in business.
For business, a metric that can be used to estimate how much more profit can be earned is more practical. CausalLift will estimate how much conversion rate (the proportion of people who took desired action such as buying a product) will increase.


### What parameaters are available for CausalLift?

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


### What was the tested environment for CausalLift?

- Google Colaboratory 


### To-dos:
- Improve documentation
- Add examples of applying uplift modeling to publicly available datasets 
- Add visualization of the uplift model
- Support for other classification models other than XGBoost to predict outcome
- Support for other classification models other than Logistic Regression to estimate propensity score

Any feedback, suggestions, pull requests to enhance documentation, usability, and features are welcomed!


### References:
- Gutierrez, P. and G´erardy, J. Causal inference and uplift modelling: A review of the literature. In International Conference on Predictive Applications and APIs, pages 1–13, 2017.

- Yi, R. and Frost, W. (n.d.). Pylift: A Fast Python Package for Uplift Modeling. Retrieved April 3, 2019, from https://tech.wayfair.com/2018/10/pylift-a-fast-python-package-for-uplift-modeling/


### About author 

Yusuke Minami

- https://github.com/Minyus
- https://www.linkedin.com/in/yusukeminami/
- https://twitter.com/Minyus86


### License

BSD 2-clause License (see https://github.com/Minyus/causallift/blob/master/LICENSE).

