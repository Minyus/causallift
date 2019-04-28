import unittest
from causallift import utils
import pandas as pd
import numpy as np
import random


class UtilsTest(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_get_cols_features_should_return_feature_columns_excluding_default_non_feature(self):
        df = pd.DataFrame(data=np.random.rand(3, 6), columns=['var1', 'var2', 'var3', 'Treatment', 'Outcome', 'Propensity'])

        result = utils.get_cols_features(df)

        self.assertEqual(['var1', 'var2', 'var3'], result)

    def test_get_cols_features_should_return_feature_columns_excluding_non_default_non_feature(self):
        df = pd.DataFrame(data=np.random.rand(3, 6), columns=['var1', 'var2', 'var3', 'MarketedTo', 'Outcome', 'Probability'])

        result = utils.get_cols_features(df, non_feature_cols=['MarketedTo', 'Outcome', 'Probability'])

        self.assertEqual(['var1', 'var2', 'var3'], result)

    def test_concat_train_test_should_concatnate_both_sets_into_series_with_keys(self):
        train_df = pd.DataFrame(data=np.random.rand(3, 3), columns=['var1', 'var2', 'var3'])
        test_df = pd.DataFrame(data=np.random.rand(3, 3), columns=['var1', 'var2', 'var3'])

        result = utils.concat_train_test(train=train_df, test=test_df)

        pd.testing.assert_series_equal(pd.Series(train_df), result.xs('train'))
        pd.testing.assert_series_equal(pd.Series(test_df), result.xs('test'))

    def test_concat_train_test_df_should_concatnate_both_sets_into_frames_with_keys(self):
        train_df = pd.DataFrame(data=np.random.rand(3, 3), columns=['var1', 'var2', 'var3'])
        test_df = pd.DataFrame(data=np.random.rand(3, 3), columns=['var1', 'var2', 'var3'])

        result = utils.concat_train_test_df(train=train_df, test=test_df)

        pd.testing.assert_frame_equal(train_df, result.xs('train'))
        pd.testing.assert_frame_equal(test_df, result.xs('test'))

    def test_len_t_should_return_the_number_of_records_where_treatment_equals_1(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=['var1', 'var2'])
        df['Treatment'] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df['Treatment'] == 1].shape[0]
        result = utils.len_t(df)

        self.assertEqual(length, result)

    def test_len_t_should_return_the_number_of_records_where_treatment_equals_0(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=['var1', 'var2'])
        df['Treatment'] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df['Treatment'] == 0].shape[0]
        result = utils.len_t(df, treatment=0)

        self.assertEqual(length, result)

    def test_len_t_should_return_the_number_of_records_where_treatment_equals_0_and_treatment_col_is_not_default(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=['var1', 'var2'])
        df['MarketedTo'] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df['MarketedTo'] == 0].shape[0]
        result = utils.len_t(df, treatment=0, col_treatment='MarketedTo')

        self.assertEqual(length, result)

    def test_len_o_should_return_the_number_of_records_where_outcome_is_1(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=['var1', 'var2'])
        df['Outcome'] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df['Outcome'] == 1].shape[0]
        result = utils.len_o(df)

        self.assertEqual(length, result)

    def test_len_o_should_return_the_number_of_records_where_outcome_is_0(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=['var1', 'var2'])
        df['Outcome'] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df['Outcome'] == 0].shape[0]
        result = utils.len_o(df, outcome=0)

        self.assertEqual(length, result)

    def test_len_o_should_return_the_number_of_records_where_outcome_equals_0_and_outcome_col_is_not_default(self):
        df = pd.DataFrame(data=np.random.rand(6, 2), columns=['var1', 'var2'])
        df['Result'] = [random.sample(range(2), 1)[0] for i in range(6)]

        length = df[df['Result'] == 0].shape[0]
        result = utils.len_o(df, outcome=0, col_outcome='Result')

        self.assertEqual(length, result)
if __name__ == '__main__':
    unittest.main()