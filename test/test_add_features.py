
import unittest
import pandas as pd

# Assume you have a function add_features defined in a module or script
from utils  import add_features

class TestAddFeatures(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'age'      : [59, 56, 41, 55, 65],
            'job'      : ['admin.', 'admin.', 'technician', 'services', 'admin.'],
            'marital'  : ['married', 'married', 'married', 'married', 'married'],
            'education': ['secondary', 'secondary', 'secondary', 'secondary', 'tertiary'],
            'default'  : ['no', 'no', 'no', 'no', 'no'],
            'balance'  : [2343, 45, 1270, 2476, 184],
            'housing'  : ['yes', 'no', 'yes', 'yes', 'no'],
            'loan'     : ['no', 'no', 'no', 'no', 'no'],
            'contact'  : ['unknown', 'unknown', 'unknown', 'unknown', 'unknown'],
            'day'      : [5, 5, 5, 5, 5],
            'month'    : ['may', 'may', 'may', 'may', 'may'],
            'duration' : [1042, 1467, 1389, 579, 673],
            'campaign' : [1, 1, 1, 1, 2],
            'pdays'    : [-1, -1, -1, -1, -1],
            'previous' : [0, 0, 0, 0, 0],
            'poutcome' : ['unknown', 'unknown', 'unknown', 'unknown', 'unknown'],
            'deposit'  : ['yes', 'yes', 'yes', 'yes', 'yes']
        })

    def test_add_features(self):
        # Create a copy of the original DataFrame
        df_copy = self.df.copy()

        # Apply the add_features function
        result_df = add_features(df_copy)

        # Check if the new columns are present in the DataFrame
        self.assertTrue('is_elderly' in result_df.columns)
        self.assertTrue('has_housing_loan' in result_df.columns)
        self.assertTrue('is_married' in result_df.columns)
        self.assertTrue('has_previous_contact' in result_df.columns)
        self.assertTrue('is_tertiary_educated' in result_df.columns)
        self.assertTrue('is_admin_job' in result_df.columns)
        self.assertTrue('has_default' in result_df.columns)
        self.assertTrue('is_month_may' in result_df.columns)
        self.assertTrue('is_loan' in result_df.columns)
        self.assertTrue('campaign_greater_than_1' in result_df.columns)

        # Check if the values in the new columns are as expected
        self.assertEqual(result_df['is_elderly'].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(result_df['has_housing_loan'].tolist(), [1, 0, 1, 1, 0])
        self.assertEqual(result_df['is_married'].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(result_df['has_previous_contact'].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(result_df['is_tertiary_educated'].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(result_df['is_admin_job'].tolist(), [1, 1, 0, 0, 1])
        self.assertEqual(result_df['has_default'].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(result_df['is_month_may'].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(result_df['is_loan'].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(result_df['campaign_greater_than_1'].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(result_df.columns.tolist(), ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome','deposit', 'is_elderly', 'has_housing_loan', 'is_married',
       'has_previous_contact', 'is_tertiary_educated', 'is_admin_job',
       'has_default', 'is_month_may', 'is_loan', 'campaign_greater_than_1'])
        
        # Add more assertions for other columns as needed

if __name__ == '__main__':
    unittest.main()
