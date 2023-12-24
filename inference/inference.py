import pickle
import pandas as pd


with open("preprocessor.pickle","rb") as file:
    loaded_prep = pickle.load(file)

test_df = pd.DataFrame([{'age': 32,
 'job': 'management',
 'marital': 'married',
 'education': 'tertiary',
 'default': 'no',
 'balance': 1500,
 'housing': 'no',
 'loan': 'no',
 'contact': 'cellular',
 'day': 28,
 'month': 'jan',
 'duration': 458,
 'campaign': 2,
 'pdays': -1,
 'previous': 0,
 'poutcome': 'unknown',
 'is_elderly': 1,
 'has_housing_loan': 0,
 'is_married': 1,
 'has_previous_contact': 1,
 'is_tertiary_educated': 1,
 'is_admin_job': 0,
 'has_default': 0,
 'is_month_may': 0,
 'is_loan': 0,
 'campaign_greater_than_1': 1}])
assert loaded_prep.transform(test_df).shape[1] == 61 , "features shapes mismatch, check your shapes"

with open("pipeline.pickle","rb") as file:
    loaded_pipeline = pickle.load(file)

result = loaded_pipeline.predict_proba(loaded_prep.transform(test_df))[:,1]
print(result[0])
