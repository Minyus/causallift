from easydict import EasyDict


def parameters_():
    args = EasyDict()
    args.cols_features = None
    args.col_treatment = 'Treatment'
    args.col_outcome = 'Outcome'
    args.col_propensity = 'Propensity'
    args.col_cate = 'CATE'
    args.col_recommendation = 'Recommendation'
    args.min_propensity = 0.01
    args.max_propensity = 0.99
    args.random_state = 0
    args.verbose = 2
    args.uplift_model_params = {
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
    args.enable_ipw = True
    args.propensity_model_params = \
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
    args.cv = 3
    args.runner = None # 'SequentialRunner' # ''ParallelRunner' # None
    args.run_only_missing = True
    return args