from easydict import EasyDict
from kedro.io import MemoryDataSet, PickleLocalDataSet, CSVLocalDataSet


def datasets_():
    datasets = EasyDict()

    # datasets.args_raw = CSVLocalDataSet(filepath='../data/01_raw/args_raw.csv', version=None)
    # datasets.train_df = CSVLocalDataSet(filepath='../data/01_raw/train_df.csv', version=None)
    # datasets.test_df = CSVLocalDataSet(filepath='../data/01_raw/test_df.csv', version=None)
    # datasets.args = CSVLocalDataSet(filepath='../data/08_reporting/args.csv', version=None)
    # datasets.df_00 = MemoryDataSet()
    # datasets.df_01 = MemoryDataSet()
    datasets.treated__model = PickleLocalDataSet(filepath='../data/06_models/treated__model.pickle', version=None)
    datasets.treated__eval_df = PickleLocalDataSet(filepath='../data/06_models/treated__eval_df.pickle', version=None)
    datasets.untreated__model = PickleLocalDataSet(filepath='../data/06_models/untreated__model.pickle', version=None)
    datasets.untreated__eval_df = PickleLocalDataSet(filepath='../data/06_models/untreated__eval_df.pickle', version=None)
    # datasets.treatment_fractions = MemoryDataSet()
    # datasets.treated__proba = MemoryDataSet()
    # datasets.untreated__proba = MemoryDataSet()
    # datasets.cate_estimated = MemoryDataSet()
    # datasets.df_02 = MemoryDataSet()
    # datasets.df_03 = CSVLocalDataSet(filepath='../data/07_model_output/df.csv', version=None)
    # datasets.treated__sim_eval_df = MemoryDataSet()
    # datasets.untreated__sim_eval_df = MemoryDataSet()
    # datasets.estimated_effect_df = CSVLocalDataSet(filepath='../data/08_reporting/estimated_effect_df.csv', version=None)

    return datasets
