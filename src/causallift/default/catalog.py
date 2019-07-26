from kedro.io import MemoryDataSet, PickleLocalDataSet, CSVLocalDataSet, JSONLocalDataSet


def datasets_():
    datasets = dict(
        # args_raw = CSVLocalDataSet(filepath='../data/01_raw/args_raw.csv', version=None),
        # train_df = CSVLocalDataSet(filepath='../data/01_raw/train_df.csv', version=None),
        # test_df = CSVLocalDataSet(filepath='../data/01_raw/test_df.csv', version=None),
        # args = JSONLocalDataSet(filepath='../data/08_reporting/args.csv', version=None),
        # df_00 = MemoryDataSet(),
        propensity_model  = PickleLocalDataSet(
            filepath='../data/06_models/propensity_model.pickle',
            version=None
        ),
        df_01 = MemoryDataSet(),
        models_dict = PickleLocalDataSet(
            filepath='../data/06_models/models_dict.pickle',
            version=None
        ),
        # treatment_fractions = MemoryDataSet(),
        # treated__proba = MemoryDataSet(),
        # untreated__proba = MemoryDataSet(),
        # cate_estimated = MemoryDataSet(),
        # df_02 = MemoryDataSet(),
        df_03 = CSVLocalDataSet(
            filepath='../data/07_model_output/df.csv',
            load_args=dict(index_col=['partition', 'index'], float_precision='high'),
            save_args=dict(index=True, float_format='%.16e'),
            version=None,
        ),
        # treated__sim_eval_df = MemoryDataSet(),
        # untreated__sim_eval_df = MemoryDataSet(),
        estimated_effect_df = CSVLocalDataSet(
            filepath='../data/08_reporting/estimated_effect_df.csv',
            version=None,
        ),
    )
    return datasets
