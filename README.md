# CausalLift: Python package for Uplift Modeling in real-world business; applicable for both A/B testing and observational data

[![PyPI version](https://badge.fury.io/py/causallift.svg)](https://badge.fury.io/py/causallift)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD-yellow.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minyus/causallift/blob/master/examples/CausalLift_example.ipynb)

If you are simply building a Machine Learning model and executing promotion campaigns to the customers who are predicted to buy a product, for example, it is not efficient.

There are some customers who will buy a product anyway even without promotion campaigns (called "Sure things").

It is even possible that the campaign triggers some customers to churn (called "Do Not Disturbs" or "Sleeping Dogs").

The solution is Uplift Modeling.

### What is Uplift Modeling?


Uplift Modeling is a Machine Learning technique to find which customers (individuals) should be targeted ("treated") and which customers should not be targeted. 

Applications of Uplift Modeling for business include:
- Increase revenue by finding which customers should be targeted for advertising/marketing campaigns and which customers should not. 
- Retain revenue by finding which customers should be contacted to prevent churn and which customers should not. 

More specifically, Uplift Modeling estimates uplift scores (a.k.a. CATE: Conditional Average Treatment Effect or ITE: Individual Treatment Effect). 


### What is uplift score?

Uplift score is how much the estimated conversion rate will increase by the campaign.

Suppose you are in charge of a marketing camppaign to sell a product, and the estimated conversion rate (probability to buy a product) of a customer if targetted is 50 % and the estimated conversion rate if not targetted is 40 %, then the uplift score of the customer is (50 - 40) = +10 % points. 

Likewise, suppose the estimated conversion rate if targeted is 20 % and the estimated conversion rate if not targetted is 80%, the uplift score is (20 - 80) = -60 % points (negative value).

The range of uplift scores is between -100 and +100 % points (-1 and +1).

It is recommended to target customers with high uplift scores and avoid customers with negative uplift scores to optimize the marketing campaign.


### What is special about "CausalLift" package?

- CausalLift works with both A/B testing results and observational datasets. 
- CausalLift can output intuitive metrics for evaluation.

### Why CausalLift was developed?

In a word, to use for real-world business.

- Existing packages for Uplift Modeling assumes the dataset is from A/B Testing (a.k.a. Randomized Controlled Trial). In real-world business, however, observational datasets in which treatment (campaign) targets were not chosen randomly are more common especially in the early stage of evidence-based decision making. CausalLift supports observational datasets using a basic methodology in Causal Inference called "Inverse Probability Weighting" based on assumption that propensity to be treated can be inferred from the available features.

- There are 2 challenges of Uplift Modeling; explainability of the model and evaluation. CausalLift utilizes a basic methodology of Uplift Modeling called Two Models approach (training 2 models independently for treated and untreated samples to compute the CATE (Conditional Average Treatment Effects) or uplift scores) to address these challenges.

	- [Explainability of the model] Since it is relatively simple, it is less challenging to explain how it works to stakeholders in business.

	- [Explainability of evaluation] To evaluate Uplift Modeling, metrics such as Qini and AUUC (Area Under the Uplift Curve) are used in research, but these metrics are difficult to explain to the stakeholders. For business, a metric that can estimate how much more profit can be earned is more practical. Since CausalLift adopted the Two-Model approach, the 2 models can be reused to simulate the outcome of following the recommendation by the Uplift Model and can estimate how much conversion rate (the proportion of people who took desired action such as buying a product) will increase using the uplift model.

### What kind of data can be fed to CausalLift?
Table data including the following columns:

- Features 
	- a.k.a independent variables, explanatory variables, covariates
	- e.g. customer gender, age range, etc.
	- Note: Categorical variables need to be one-hot coded so propensity can be estimated using logistic regression. [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) can be used.
- Outcome: binary (0 or 1)
	- a.k.a dependent variable, target variable, label
	- e.g. whether the customer bought a product, clicked a link, etc.
- Treatment: binary (0 or 1)
	- a variable you can control and want to optimize for each individual (customer) 
	- a.k.a intervention
	- e.g. whether an advertising campaign was executed, whether a discount was offered, etc.
	- Note: if you cannot find a treatment column, you may need to ask stakeholders to get the data, which might take hours to years.
- [Optional] Propensity: continuous between 0 and 1
	- propensity (or probability) to be treated for observational datasets (not needed for A/B Testing results)
	- If not provided, CausalLift can estimate from the features using logistic regression. 


<img src="readme_images/Example_table_data.png">
<p align="center">
	Example table data
</p>



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
https://colab.research.google.com/github/Minyus/causallift/blob/master/examples/CausalLift_example.ipynb
) available in Google Colab (free cloud CPU/GPU) environment.

To run the code, navigate to "Runtime" >> "Run all".

To download the notebook file, navigate to "File" >> "Download .ipynb".

Here are the basic steps to use.

```python
""" Step 0. Import CausalLift 
"""

from causallift import CausalLift

""" Step 1. Feed datasets and optionally compute estimated propensity scores 
using logistic regression if set enable_ipw = True.
"""

cl = CausalLift(train_df, test_df, enable_ipw=True)

""" Step 2. Train 2 classification models (XGBoost) for treated and untreated 
samples independently and compute estimated CATE (Conditional Average Treatment 
Effect), ITE (Individual Treatment Effect), or uplift score. 
"""

train_df, test_df = cl.estimate_cate_by_2_models()

""" Step 3. Estimate how much conversion rate will increase by selecting treatment 
(campaign) targets as recommended by the uplift modeling. 
"""

estimated_effect_df = cl.estimate_recommendation_impact()
```

<img src="readme_images/CausalLift_flow_diagram.png">
<p align="center">
	CausalLift flow diagram
</p>

### What parameters are available for CausalLift?

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


### What was the tested environment of CausalLift?

- Google Colab (Python 3.6.7)


### Related Python packages

- ["pylift"](https://github.com/wayfair/pylift)
[[documentation]](https://pylift.readthedocs.io/en/latest/)

	Uplift Modeling based on Transformed Outcome method for A/B Testing data and visualization of metrics such as Qini.
	
- ["EconML" (ALICE: Automated Learning and Intelligence for Causation and Economics)](https://github.com/Microsoft/EconML) 
[[documentation]](https://econml.azurewebsites.net/index.html)

	Several advanced methods to estimate CATE from observational data.
	
- ["DoWhy"](https://github.com/Microsoft/dowhy)
[[documentation]](https://causalinference.gitlab.io/dowhy/)

	Visualization of steps in Causal Inference for observational data.
	
- ["pymatch"](https://github.com/benmiroglio/pymatch)

	Propensity Score Matching for observational data.

- ["Ax"](https://github.com/facebook/Ax) 
[[documentation]](https://ax.dev/)

	Platform for adaptive experiments, powered by BoTorch, a library built on PyTorch

### Related R packages

- ["uplift"](https://cran.r-project.org/web/packages/uplift/index.html)

	Uplift Modeling.
	
- ["tools4uplift"](https://cran.r-project.org/web/packages/tools4uplift/index.html) [[paper]](https://arxiv.org/abs/1901.10867)

	Uplift Modeling and utility tools for quantization of continuous variables, visualization of metrics such as Qini, and automatic feature selection.

- ["matching"](https://cran.r-project.org/web/packages/Matching/index.html)

	Propensity Score Matching for observational data.

- ["CausalImpact"](https://cran.r-project.org/web/packages/CausalImpact/index.html) [[documentation]](https://google.github.io/CausalImpact/CausalImpact.html)

	Causal inference using Bayesian structural time-series models 


### References

- Gutierrez, Pierre. and G´erardy, Jean-Yves. Causal inference and uplift modelling: A review of the literature. In International Conference on Predictive Applications and APIs, pages 1–13, 2017.

- Athey, Susan and Imbens, Guido W. Machine learning methods for estimating heterogeneous causal effects. Stat, 2015.

- Yi, Robert. and Frost, Will. (n.d.). Pylift: A Fast Python Package for Uplift Modeling. Retrieved April 3, 2019, from https://tech.wayfair.com/2018/10/pylift-a-fast-python-package-for-uplift-modeling/


### Introductive resources about Uplift Modeling

- [[Medium article] Uplift Models for better marketing campaigns (Part 1)](
https://medium.com/@abhayspawar/uplift-models-for-better-marketing-campaigns-part-1-b491292e4c80
)
- [[Medium article] Simple Machine Learning Techniques To Improve Your Marketing Strategy: Demystifying Uplift Models](
https://medium.com/datadriveninvestor/simple-machine-learning-techniques-to-improve-your-marketing-strategy-demystifying-uplift-models-dc4fb3f927a2
)
- [[Wikipedia] Uplift_modelling](
https://en.wikipedia.org/wiki/Uplift_modelling
)

### License

[BSD 2-clause License](https://github.com/Minyus/causallift/blob/master/LICENSE).


### To-dos

- Improve documentation (using Sphinx)
- Add examples of applying uplift modeling to more publicly available datasets (such as [Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data) as [pymatch](https://github.com/benmiroglio/pymatch) did.
- Clarify the model summary output including visualization
- Support for classification models other than XGBoost (Random Forest, LightGBM, etc.) to predict outcome
- Support for classification models other than Logistic Regression to estimate propensity scores which would need calibration
- Support for multiple treatments

Any feedback, suggestions, pull requests to enhance/correct documentation, usability, and features are welcomed!
Separate pull requests for each improvement are appreciated rather than a single big pull request.

If you could write a review about CausalLift in any natural languages (English, Chinese, Japanese, etc.) or implement similar features in any programming languages (R, SAS, etc.), please let me know. I will add the link here.

### Keywords to search

[English] Causal Inference, Counterfactual, Propensity Score, Econometrics

[中文] 因果推论, 反事实, 倾向评分, 计量经济学

[日本語] 因果推論, 反事実, 傾向スコア, 計量経済学

### Article about CausalList in Japanese
- https://qiita.com/Minyus86/items/07ce57a8bddc49c2bbf5

### About author 

Yusuke Minami

- https://github.com/Minyus
- https://www.linkedin.com/in/yusukeminami/
- https://twitter.com/Minyus86
