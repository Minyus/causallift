import unittest
from causallift import utils
import pandas as pd
import numpy as np


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

if __name__ == '__main__':
    unittest.main()