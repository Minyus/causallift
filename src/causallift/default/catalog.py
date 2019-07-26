from kedro.io import PickleLocalDataSet, CSVLocalDataSet, JSONLocalDataSet


def datasets_():
    datasets = dict(
        # args_raw = CSVLocalDataSet(filepath='../data/01_raw/args_raw.csv', version=None),
        # train_df = CSVLocalDataSet(filepath='../data/01_raw/train_df.csv', version=None),
        # test_df = CSVLocalDataSet(filepath='../data/01_raw/test_df.csv', version=None),
        propensity_model  = PickleLocalDataSet(
            filepath='../data/06_models/propensity_model.pickle',
            version=None
        ),
        models_dict = PickleLocalDataSet(
            filepath='../data/06_models/models_dict.pickle',
            version=None
        ),
        df_03 = CSVLocalDataSet(
            filepath='../data/07_model_output/df.csv',
            load_args=dict(index_col=['partition', 'index'], float_precision='high'),
            save_args=dict(index=True, float_format='%.16e'),
            version=None,
        ),
        treated__sim_eval_df = CSVLocalDataSet(
            filepath='../data/08_reporting/treated__sim_eval_df.csv',
            version=None,
        ),
        untreated__sim_eval_df = CSVLocalDataSet(
            filepath='../data/08_reporting/untreated__sim_eval_df.csv',
            version=None,
        ),
        estimated_effect_df = CSVLocalDataSet(
            filepath='../data/08_reporting/estimated_effect_df.csv',
            version=None,
        ),
    )
    return datasets
