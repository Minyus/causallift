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

if __name__ == '__main__':
    unittest.main()