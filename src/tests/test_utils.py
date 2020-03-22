from pathlib import Path
import random
import sys
import unittest

from easydict import EasyDict
import numpy as np
import pandas as pd

from causallift import utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_cols_features_should_return_feature_columns_excluding_default_non_feature(
        self,
    ):
        df = pd.DataFrame(
            data=np.random.rand(3, 6),
            columns=["var1", "var2", "var3", "Treatment", "Outcome", "Propensity"],
        )

        result = utils.get_cols_features(df)

        self.assertEqual(["var1", "var2", "var3"], result)

    def test_get_cols_features_should_return_feature_columns_excluding_non_default_non_feature(
        self,
    ):
        df = pd.DataFrame(
            data=np.random.rand(3, 6),
            columns=["var1", "var2", "var3", "MarketedTo", "Outcome", "Probability"],
        )

        result = utils.get_cols_features(
            df, non_feature_cols=["MarketedTo", "Outcome", "Probability"]
        )

        self.assertEqual(["var1", "var2", "var3"], result)

    def test_concat_train_test_should_concatnate_both_sets_into_series_with_keys(self):
        args = EasyDict(dict(index_name="index", partition_name="partition"))

        train_array = np.random.rand(3)
        test_array = np.random.rand(3)
        result = utils.concat_train_test(args=args, train=train_array, test=test_array)

        np.testing.assert_array_equal(train_array, result.xs("train"))
        np.testing.assert_array_equal(test_array, result.xs("test"))

    def test_concat_train_test_df_should_concatnate_both_sets_into_frames_with_keys(
        self,
    ):
        args = EasyDict(dict(index_name="index", partition_name="partition"))
        train_df = pd.DataFrame(
            data=np.random.rand(3, 3), columns=["var1", "var2", "var3"]
        )
        train_df.index.name = args.index_name
        test_df = pd.DataFrame(
            data=np.random.rand(3, 3), columns=["var1", "var2", "var3"]
        )
        test_df.index.name = args.index_name

        result = utils.concat_train_test_df(args=args, train=train_df, test=test_df)

        pd.testing.assert_frame_equal(train_df, result.xs("train"))
        pd.testing.assert_frame_equal(test_df, result.xs("test"))

    def test_len_t_should_return_the_number_of_records_where_treatment_equals_1(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=["var1", "var2"])
        df["Treatment"] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df["Treatment"] == 1].shape[0]
        result = utils.len_t(df)

        self.assertEqual(length, result)

    def test_len_t_should_return_the_number_of_records_where_treatment_equals_0(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=["var1", "var2"])
        df["Treatment"] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df["Treatment"] == 0].shape[0]
        result = utils.len_t(df, treatment=0)

        self.assertEqual(length, result)

    def test_len_t_should_return_the_number_of_records_where_treatment_equals_0_and_treatment_col_is_not_default(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=["var1", "var2"])
        df["MarketedTo"] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df["MarketedTo"] == 0].shape[0]
        result = utils.len_t(df, treatment=0, col_treatment="MarketedTo")

        self.assertEqual(length, result)

    def test_len_o_should_return_the_number_of_records_where_outcome_is_1(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=["var1", "var2"])
        df["Outcome"] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df["Outcome"] == 1].shape[0]
        result = utils.len_o(df)

        self.assertEqual(length, result)

    def test_len_o_should_return_the_number_of_records_where_outcome_is_0(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=["var1", "var2"])
        df["Outcome"] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df["Outcome"] == 0].shape[0]
        result = utils.len_o(df, outcome=0)

        self.assertEqual(length, result)

    def test_len_o_should_return_the_number_of_records_where_outcome_equals_0_and_outcome_col_is_not_default(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=["var1", "var2"])
        df["Result"] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df["Result"] == 0].shape[0]
        result = utils.len_o(df, outcome=0, col_outcome="Result")

        self.assertEqual(length, result)

    def test_len_to_should_return_the_number_of_records_where_outcome_and_treatment_is_1(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["Outcome"] = [random.sample(range(2), 1)[0] for i in range(12)]
        df["Treatment"] = [random.sample(range(2), 1)[0] for i in range(12)]

        length = df[(df["Treatment"] == 1) & (df["Outcome"] == 1)].shape[0]
        result = utils.len_to(df)

        self.assertEqual(length, result)

    def test_len_to_should_return_the_number_of_records_where_outcome_and_treatment_are_different(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["Outcome"] = [random.sample(range(2), 1)[0] for i in range(12)]
        df["Treatment"] = [random.sample(range(2), 1)[0] for i in range(12)]

        length = df[(df["Treatment"] == 1) & (df["Outcome"] == 0)].shape[0]
        result = utils.len_to(df, outcome=0)

        self.assertEqual(length, result)

    def test_len_to_should_return_the_number_of_records_where_outcome_and_treatment_are_different_with_custom_column_names(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["result"] = [random.sample(range(2), 1)[0] for i in range(12)]
        df["marketed_to"] = [random.sample(range(2), 1)[0] for i in range(12)]

        length = df[(df["marketed_to"] == 1) & (df["result"] == 0)].shape[0]
        result = utils.len_to(
            df, outcome=0, col_outcome="result", col_treatment="marketed_to"
        )

        self.assertEqual(length, result)

    def test_treatment_fraction_should_compute_percentage_of_treated(self):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["Treatment"] = [random.sample(range(2), 1)[0] for i in range(12)]

        value = len(df[df["Treatment"] == 1]) / len(df)
        result = utils.treatment_fraction_(df)

        self.assertEqual(value, result)

    def test_treatment_fraction_should_compute_percentage_of_treated_with_custom_name(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["marketed"] = [random.sample(range(2), 1)[0] for i in range(12)]

        value = len(df[df["marketed"] == 1]) / len(df)
        result = utils.treatment_fraction_(df, col_treatment="marketed")

        self.assertEqual(value, result)

    def test_outcome_fraction_should_compute_percentage_of_positive_outcome(self):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["Outcome"] = [random.sample(range(2), 1)[0] for i in range(12)]

        value = len(df[df["Outcome"] == 1]) / len(df)
        result = utils.outcome_fraction_(df)

        self.assertEqual(value, result)

    def test_outcome_fraction_should_compute_percentage_of_positive_outcome_with_custom_name(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["result"] = [random.sample(range(2), 1)[0] for i in range(12)]

        value = len(df[df["result"] == 1]) / len(df)
        result = utils.outcome_fraction_(df, col_outcome="result")

        self.assertEqual(value, result)

    def test_overall_uplift_gain_should_compute_uplift_for_sure_things(self):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["Outcome"] = [random.sample(range(2), 1)[0] for i in range(12)]
        df["Treatment"] = [random.sample(range(2), 1)[0] for i in range(12)]

        no_treated_positive_outcome = df[
            (df["Treatment"] == 1) & (df["Outcome"] == 1)
        ].shape[0]
        no_not_treated_positive_outcome = df[
            (df["Treatment"] == 0) & (df["Outcome"] == 1)
        ].shape[0]
        no_treated = df[df["Treatment"] == 1].shape[0]
        no_not_treated = df[df["Treatment"] == 0].shape[0]

        gain = (no_treated_positive_outcome / no_treated) - (
            no_not_treated_positive_outcome / no_not_treated
        )
        result = utils.overall_uplift_gain_(df)

        self.assertEqual(gain, result)

    def test_overall_uplift_gain_should_compute_uplift_for_sure_things_with_custom_colum_names(
        self,
    ):
        df = pd.DataFrame(data=np.random.rand(12, 2), columns=["var1", "var2"])
        df["Result"] = [random.sample(range(2), 1)[0] for i in range(12)]
        df["Contacted"] = [random.sample(range(2), 1)[0] for i in range(12)]

        no_treated_positive_outcome = df[
            (df["Contacted"] == 1) & (df["Result"] == 1)
        ].shape[0]
        no_not_treated_positive_outcome = df[
            (df["Contacted"] == 0) & (df["Result"] == 1)
        ].shape[0]
        no_treated = df[df["Contacted"] == 1].shape[0]
        no_not_treated = df[df["Contacted"] == 0].shape[0]

        gain = (no_treated_positive_outcome / no_treated) - (
            no_not_treated_positive_outcome / no_not_treated
        )
        result = utils.overall_uplift_gain_(
            df, col_treatment="Contacted", col_outcome="Result"
        )

        self.assertEqual(gain, result)


if __name__ == "__main__":
    unittest.main()
